import React, { useCallback, useMemo, useState } from "react";
import { Dimensions, RefreshControl, ScrollView, StyleSheet, View } from "react-native";
import { ActivityIndicator, Text } from "react-native-paper";
import { BarChart, LineChart } from "react-native-chart-kit";
import ConnectionBanner from "../components/ConnectionBanner";
import TabChipRow from "../components/TabChipRow";
import { useApiContext } from "../context/ApiContext";
import { ANALYTICS_TABS, ZONE_COLORS } from "../shared/constants";
import { colors, layout, spacing } from "../theme";
import {
  filterAnalysis,
  filterByZoneField,
  filterCrowdZones,
  filterZones,
} from "../utils/zoneFilter";

const chartWidth = Dimensions.get("window").width - spacing.md * 2;

const chartConfig = {
  backgroundColor: colors.surface,
  backgroundGradientFrom: colors.surfaceAlt,
  backgroundGradientTo: colors.surface,
  backgroundGradientFromOpacity: 0.96,
  backgroundGradientToOpacity: 0.96,
  decimalPlaces: 0,
  color: (opacity = 1) => `rgba(248, 250, 252, ${opacity})`,
  labelColor: (opacity = 1) => `rgba(226, 232, 240, ${opacity * 0.85})`,
  propsForDots: {
    r: "4",
    strokeWidth: "2",
    stroke: colors.surface,
    fill: colors.primary,
  },
  propsForBackgroundLines: {
    stroke: "rgba(148,163,184,0.14)",
    strokeDasharray: "4",
  },
  fillShadowGradient: colors.primary,
  fillShadowGradientOpacity: 0.35,
  barPercentage: 0.72,
  useShadowColorFromDataset: true,
};

function formatTimelineLabel(point) {
  const raw = point?.timestamp ?? point?.ts ?? point?.time;
  if (!raw) return "—";
  const ms = typeof raw === "number" && raw < 1e12 ? raw * 1000 : raw;
  const date = new Date(ms);
  if (Number.isNaN(date.getTime())) return "—";
  return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

function normalizeCrowdZones(crowdStatus) {
  const raw = crowdStatus?.zones ?? crowdStatus?.items;
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  return Object.entries(raw).map(([zoneId, zone]) => ({
    zone: zoneId,
    id: zoneId,
    ...zone,
  }));
}

function getCrowdRiskLabel(zones) {
  const crowded = zones.filter((zone) => zone.exceeded || zone.status === "crowded").length;
  if (crowded >= 3) return { label: "High", color: colors.danger };
  if (crowded === 2) return { label: "Medium", color: colors.warning };
  if (crowded === 1) return { label: "Elevated", color: colors.primary };
  return { label: "Normal", color: colors.success };
}

function getTrendLabel(delta) {
  if (delta > 15) return { label: "Sharp rise", color: colors.danger };
  if (delta > 7) return { label: "Rising", color: colors.warning };
  if (delta >= -7) return { label: "Stable", color: colors.primary };
  return { label: "Improving", color: colors.success };
}

function formatUpdatedAt(timestamp) {
  if (!timestamp) return "—";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function parseDateMs(raw) {
  if (!raw) return 0;
  const ms = typeof raw === "number" && raw < 1e12 ? raw * 1000 : raw;
  const date = new Date(ms);
  return Number.isNaN(date.getTime()) ? 0 : date.getTime();
}

function getZoneRiskBadge(status) {
  const normalized = String(status || "safe").toLowerCase();
  if (normalized === "crowded" || normalized === "high") return { label: "Crowded", color: colors.danger };
  if (normalized === "warning" || normalized === "elevated" || normalized === "medium") return { label: "Warning", color: colors.warning };
  return { label: "Safe", color: colors.success };
}

export default function AnalyticsScreen() {
  const { baseUrl, api, connected, assignedZones } = useApiContext();
  const [activeTab, setActiveTab] = useState("overview");
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [crowdStatus, setCrowdStatus] = useState(null);
  const [crowdAlerts, setCrowdAlerts] = useState([]);
  const [crowdLastUpdated, setCrowdLastUpdated] = useState(null);
  const [responseTimes, setResponseTimes] = useState(null);
  const [personCountZone, setPersonCountZone] = useState(null);
  const [personZoneOptions, setPersonZoneOptions] = useState([]);
  const [loadedTabs, setLoadedTabs] = useState({});

  const personZoneTabs = useMemo(
    () =>
      personZoneOptions.map((zone) => ({
        id: String(zone.id),
        label: zone.name || `Zone ${zone.id}`,
      })),
    [personZoneOptions]
  );

  const loadPersonCountTimeline = useCallback(
    async (zone) => {
      if (!zone) {
        setPersonCountZone(null);
        setTimeline({ timeline: [] });
        return;
      }
      setPersonCountZone(zone);
      setTimeline(await api.timeline(zone.id));
    },
    [api]
  );

  const loadTab = useCallback(
    async (tabId) => {
      if (!baseUrl) return;
      setLoading(true);
      setError(null);
      try {
        if (tabId === "overview") {
          const raw = await api.analysis();
          setAnalysis(filterAnalysis(raw, assignedZones));
        } else if (tabId === "person_count") {
          const zones = await api.zones();
          const scoped = filterZones(zones?.items || [], assignedZones);
          setPersonZoneOptions(scoped);
          const activeZone =
            scoped.find((z) => String(z.id) === String(personCountZone?.id)) || scoped[0];
          await loadPersonCountTimeline(activeZone);
        } else if (tabId === "crowd") {
          const [rawStatus, alerts] = await Promise.all([api.crowdStatus(), api.crowdAlerts(20)]);
          const zones = filterCrowdZones(normalizeCrowdZones(rawStatus), assignedZones);
          const updated = {
            ...rawStatus,
            zones,
            crowded_zones_count: zones.filter((z) => z.exceeded || z.status === "crowded").length,
          };
          setCrowdStatus(updated);
          setCrowdAlerts(filterByZoneField(alerts?.alerts || [], assignedZones));
          setCrowdLastUpdated(Date.now());
        } else if (tabId === "response") {
          const raw = await api.responseTimes();
          const recent = filterByZoneField(raw?.recent || [], assignedZones);
          setResponseTimes({ ...raw, recent });
        }
        setLoadedTabs((prev) => ({ ...prev, [tabId]: true }));
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    },
    [api, assignedZones, baseUrl, loadPersonCountTimeline, personCountZone?.id]
  );

  const handlePersonZoneSelect = useCallback(
    async (zoneId) => {
      const zone = personZoneOptions.find((z) => String(z.id) === String(zoneId));
      if (!zone) return;
      setLoading(true);
      setError(null);
      try {
        await loadPersonCountTimeline(zone);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    },
    [loadPersonCountTimeline, personZoneOptions]
  );

  React.useEffect(() => {
    setLoadedTabs({});
  }, [assignedZones]);

  React.useEffect(() => {
    if (!baseUrl) return;
    if (!loadedTabs[activeTab]) {
      loadTab(activeTab);
    }
  }, [activeTab, baseUrl, loadTab, loadedTabs]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await loadTab(activeTab);
    } finally {
      setRefreshing(false);
    }
  }, [activeTab, loadTab]);

  const overviewCards = useMemo(() => {
    if (!analysis) return [];
    return [
      { label: "Total Alerts", value: analysis.alerts_total ?? 0 },
      { label: "Zones With Alerts", value: Object.keys(analysis.alerts_by_zone || {}).length },
      { label: "Detection Types", value: Object.keys(analysis.alerts_by_label || {}).length },
    ];
  }, [analysis]);

  const zoneBreakdown = useMemo(() => {
    if (!analysis?.alerts_by_zone) return [];
    return Object.entries(analysis.alerts_by_zone)
      .sort((a, b) => b[1] - a[1])
      .map(([zone, count]) => ({ zone, count }));
  }, [analysis]);

  const labelBreakdown = useMemo(() => {
    if (!analysis?.alerts_by_label) return [];
    return Object.entries(analysis.alerts_by_label)
      .sort((a, b) => b[1] - a[1])
      .map(([label, count]) => ({ label, count }));
  }, [analysis]);

  const timelineChart = useMemo(() => {
    const points = timeline?.timeline || [];
    if (!points.length) return null;
    const slice = points.slice(-8);
    const labels = slice.map(formatTimelineLabel);
    const data = slice.map((p) => Number(p.count ?? p.person_count ?? 0));
    return {
      labels: labels.length ? labels : ["—"],
      datasets: [{ data: data.length ? data : [0] }],
    };
  }, [timeline]);

  const latestPersonCount = useMemo(() => {
    const points = timeline?.timeline || [];
    if (!points.length) return null;
    return points[points.length - 1]?.count ?? 0;
  }, [timeline]);

  const crowdZones = useMemo(() => normalizeCrowdZones(crowdStatus), [crowdStatus]);

  const crowdZoneSummary = useMemo(() => {
    const zones = crowdZones || [];
    const totalPeople = zones.reduce((sum, zone) => sum + Number(zone.person_count ?? zone.count ?? 0), 0);
    const crowdedCount = zones.filter((zone) => zone.exceeded || zone.status === "crowded").length;
    const averageLoad = zones.length ? totalPeople / zones.length : 0;
    const topZone = zones
      .slice()
      .sort((a, b) => Number(b.person_count ?? b.count ?? 0) - Number(a.person_count ?? a.count ?? 0))[0];
    const risk = getCrowdRiskLabel(zones);
    const trendDelta = (zones.length ? topZone?.person_count ?? topZone?.count ?? 0 : 0) - (averageLoad || 0);
    const trend = getTrendLabel(trendDelta);
    return {
      totalPeople,
      crowdedCount,
      averageLoad: Math.round(averageLoad),
      topZone,
      risk,
      trend,
    };
  }, [crowdZones]);

  const crowdZoneProgress = useMemo(() => {
    return crowdZones.map((zone, index) => {
      const count = Number(zone.person_count ?? zone.count ?? 0);
      const threshold = Number(zone.threshold ?? zone.limit ?? 0);
      const percent = threshold > 0 ? Math.min(count / threshold, 1) : 0;
      const rawDelta = threshold > 0 ? Math.round((count / threshold) * 100) : 0;
      const riskText = rawDelta >= 100 ? "Crowded" : rawDelta >= 75 ? "High" : rawDelta >= 50 ? "Medium" : "Safe";
      const colorIndex = index % ZONE_COLORS.length;
      return {
        id: zone.zone ?? zone.id,
        label: zone.zone_name || zone.name || `Zone ${zone.zone ?? zone.id}`,
        description: zone.description || zone.zone_description || "No description available",
        count,
        threshold,
        percent,
        crowdPressure: `${rawDelta}%`,
        pressureLabel: riskText,
        status: zone.status || (threshold > 0 && count >= threshold ? "crowded" : rawDelta >= 75 ? "warning" : "safe"),
        accent: ZONE_COLORS[colorIndex] || colors.primary,
      };
    });
  }, [crowdZones]);

  const crowdAlertsByZone = useMemo(() => {
    const bucketsByZone = {};
    crowdAlerts.forEach((alert) => {
      const zoneId = String(alert.zone ?? alert.zone_id ?? "—");
      const timestamp = parseDateMs(alert.timestamp ?? alert.ts ?? alert.ts_utc);
      if (!bucketsByZone[zoneId]) {
        bucketsByZone[zoneId] = { zone: zoneId, alerts: [], count: 0 };
      }
      bucketsByZone[zoneId].count += 1;
      bucketsByZone[zoneId].alerts.push(timestamp);
    });

    return Object.values(bucketsByZone)
      .map((zone) => {
        const sorted = zone.alerts.sort((a, b) => a - b);
        const latest = sorted.slice(-6);
        let sparkline = [0, 0, 0, 0, 0, 0];
        if (latest.length) {
          const minTs = latest[0];
          const maxTs = latest[latest.length - 1] || minTs + 1;
          const span = Math.max(maxTs - minTs, 1);
          sparkline = latest.map((ts) => Math.round(((ts - minTs) / span) * 5));
        }
        return { ...zone, sorted, sparkline };
      })
      .sort((a, b) => b.count - a.count);
  }, [crowdAlerts]);

  const topCrowdAlertZones = useMemo(() => crowdAlertsByZone.slice(0, 3), [crowdAlertsByZone]);

  const crowdChart = useMemo(() => {
    if (!crowdZones.length) return null;

    const sorted = crowdZones
      .slice()
      .sort((a, b) => Number(b.person_count ?? b.count ?? 0) - Number(a.person_count ?? a.count ?? 0));

    const dataPoints = sorted.map((z) => Number(z.person_count ?? z.count ?? 0));
    const thresholdPoints = sorted.map((z) => Number(z.threshold ?? z.limit ?? 0));
    const accentColors = sorted.map((z, index) => {
      const color = ZONE_COLORS[index % ZONE_COLORS.length] || colors.primary;
      return (opacity = 1) => `rgba(${parseInt(color.slice(1, 3), 16)}, ${parseInt(color.slice(3, 5), 16)}, ${parseInt(color.slice(5, 7), 16)}, ${opacity * 0.95})`;
    });

    return {
      labels: sorted.map((z) => z.zone_name || z.name || `Zone ${z.zone || z.id}`),
      datasets: [
        {
          data: dataPoints,
          color: (opacity = 1, index) => accentColors[index](opacity),
          withCustomBarColorFromData: true,
        },
      ],
      legend: ["Current load"],
    };
  }, [crowdZones]);

  const responseRows = useMemo(() => {
    return responseTimes?.recent || responseTimes?.items || responseTimes?.responses || [];
  }, [responseTimes]);

  const responseChart = useMemo(() => {
    if (!responseRows.length) return null;
    return {
      labels: responseRows.slice(0, 6).map((r) => `Z${r.zone}`),
      datasets: [
        {
          data: responseRows.slice(0, 6).map((r) => Number(r.response_time_seconds ?? r.seconds ?? 0)),
          color: (opacity = 1) => `rgba(245, 158, 11, ${opacity})`,
        },
      ],
      legend: ["Response time (s)"],
    };
  }, [responseRows]);

  const responseSummary = useMemo(() => {
    return responseTimes?.overall || null;
  }, [responseTimes]);

  const lifeguardRows = useMemo(() => {
    return responseTimes?.by_lifeguard
      ? Object.entries(responseTimes.by_lifeguard).map(([id, stats]) => ({ id, ...stats }))
      : [];
  }, [responseTimes]);

  const responseRisk = useMemo(() => {
    const avg = responseSummary?.avg_response_time ?? 0;
    if (avg > 12) return { label: "Slow", color: colors.danger };
    if (avg > 7) return { label: "Moderate", color: colors.warning };
    return { label: "Fast", color: colors.success };
  }, [responseSummary]);

  const zoneRows = useMemo(() => {
    return responseTimes?.by_zone
      ? Object.entries(responseTimes.by_zone).map(([zone, stats]) => ({ zone, ...stats }))
      : [];
  }, [responseTimes]);

  if (!baseUrl) {
    return (
      <View style={styles.centered}>
        <Text style={styles.emptyTitle}>Configure server URL in Settings</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ConnectionBanner visible={!connected} />
      <View style={styles.pageHeader}>
        <Text style={styles.pageTitle}>Situational Awareness</Text>
        <Text style={styles.pageSubtitle}>Performance metrics, crowd insights, and zone intelligence.</Text>
      </View>
      <TabChipRow tabs={ANALYTICS_TABS} activeId={activeTab} onSelect={setActiveTab} />

      <ScrollView
        style={styles.body}
        contentContainerStyle={styles.content}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.primary} />}
      >
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionHeaderTitle}>Analytics Center</Text>
          <Text style={styles.sectionHeaderSub}>Track alerts, crowd levels, and response performance</Text>
        </View>

        <View style={styles.premiumBanner}>
          <Text style={styles.premiumBadge}>Premium Insights</Text>
          <Text style={styles.premiumText}>
            Advanced crowd monitoring and response analytics help keep your coastal team one step ahead.
          </Text>
        </View>

        {loading && !loadedTabs[activeTab] ? (
          <ActivityIndicator color={colors.primary} style={{ marginTop: spacing.lg }} />
        ) : null}

        {error ? <Text style={styles.error}>{error}</Text> : null}

        {activeTab === "overview" && overviewCards.length > 0 && (
          <>
            <View style={styles.cardGrid}>
              {overviewCards.map((card) => (
                <View key={card.label} style={styles.statCard}>
                  <Text style={styles.statValue}>{card.value}</Text>
                  <Text style={styles.statLabel}>{card.label}</Text>
                </View>
              ))}
            </View>

            {zoneBreakdown.length > 0 && (
              <View style={styles.breakdownCard}>
                <Text style={styles.breakdownTitle}>Alerts by Zone</Text>
                {zoneBreakdown.map((row) => (
                  <View key={row.zone} style={styles.breakdownRow}>
                    <Text style={styles.breakdownLabel}>Zone {row.zone}</Text>
                    <Text style={styles.breakdownValue}>{row.count}</Text>
                  </View>
                ))}
              </View>
            )}

            {labelBreakdown.length > 0 && (
              <View style={styles.breakdownCard}>
                <Text style={styles.breakdownTitle}>Alerts by Type</Text>
                {labelBreakdown.map((row) => (
                  <View key={row.label} style={styles.breakdownRow}>
                    <Text style={styles.breakdownLabel}>{row.label}</Text>
                    <Text style={styles.breakdownValue}>{row.count}</Text>
                  </View>
                ))}
              </View>
            )}
          </>
        )}

        {activeTab === "person_count" && (
          <>
            {personZoneTabs.length > 1 ? (
              <TabChipRow
                tabs={personZoneTabs}
                activeId={personCountZone ? String(personCountZone.id) : personZoneTabs[0]?.id}
                onSelect={handlePersonZoneSelect}
              />
            ) : personCountZone ? (
              <Text style={styles.sectionTitle}>
                {personCountZone.name || `Zone ${personCountZone.id}`}
              </Text>
            ) : null}
            {latestPersonCount !== null && (
              <View style={styles.statCardWide}>
                <Text style={styles.statValue}>{latestPersonCount}</Text>
                <Text style={styles.statLabel}>Current person count</Text>
              </View>
            )}
            {timelineChart ? (
              <LineChart
                data={timelineChart}
                width={chartWidth}
                height={layout.chartHeight}
                chartConfig={chartConfig}
                bezier
                style={styles.chart}
              />
            ) : (
              <Text style={styles.emptyText}>Person count timeline will appear after the zone runs for a few seconds.</Text>
            )}
          </>
        )}

        {activeTab === "crowd" && (
          <>
            {crowdStatus ? (
              <View style={styles.crowdSummaryCard}>
                <View style={styles.crowdSummaryHeader}>
                  <View>
                    <Text style={styles.crowdSummaryLabel}>Live Crowd Risk</Text>
                    <Text style={[styles.crowdRiskValue, { color: crowdZoneSummary.risk.color }]}> 
                      {crowdZoneSummary.risk.label}
                    </Text>
                  </View>
                  <View style={styles.crowdSummaryMeta}>
                    <View style={[styles.riskPill, { backgroundColor: `${crowdZoneSummary.risk.color}18`, borderColor: `${crowdZoneSummary.risk.color}44` }] }>
                      <Text style={[styles.riskPillText, { color: crowdZoneSummary.risk.color }]}> {crowdZoneSummary.crowdedCount} crowded zones </Text>
                    </View>
                    <Text style={styles.updatedAtText}>Updated {formatUpdatedAt(crowdLastUpdated)}</Text>
                  </View>
                </View>
                <View style={styles.summaryStats}>
                  <View style={styles.summaryStatItem}>
                    <Text style={styles.summaryStatLabel}>Total People</Text>
                    <Text style={styles.summaryStatValue}>{crowdZoneSummary.totalPeople}</Text>
                  </View>
                  <View style={styles.summaryStatItem}>
                    <Text style={styles.summaryStatLabel}>Average load</Text>
                    <Text style={styles.summaryStatValue}>{crowdZoneSummary.averageLoad}</Text>
                  </View>
                  <View style={styles.summaryStatItem}>
                    <Text style={styles.summaryStatLabel}>Top risk zone</Text>
                    <Text style={styles.summaryStatValue}>
                      {crowdZoneSummary.topZone?.zone_name || `Zone ${crowdZoneSummary.topZone ? (crowdZoneSummary.topZone.zone ?? crowdZoneSummary.topZone.id) : "—"}`}
                    </Text>
                  </View>
                </View>
                <View style={styles.trendRow}>
                  <Text style={styles.trendLabel}>Trend</Text>
                  <Text style={[styles.trendValue, { color: crowdZoneSummary.trend.color }]}>{crowdZoneSummary.trend.label}</Text>
                </View>
              </View>
            ) : null}

            <View style={styles.topZonesGroup}>
              {crowdZoneProgress
                .slice()
                .sort((a, b) => Number(b.count) - Number(a.count))
                .slice(0, 3)
                .map((zone) => {
                  const badge = getZoneRiskBadge(zone.status);
                  return (
                    <View key={zone.id} style={[styles.zoneCard, { borderColor: `${zone.accent}22` }]}> 
                      <View style={styles.zoneCardHeader}>
                        <Text style={styles.zoneCardLabel}>{zone.label}</Text>
                        <View style={[styles.statusBadge, { backgroundColor: `${badge.color}18`, borderColor: `${badge.color}44` }]}> 
                          <Text style={[styles.statusBadgeText, { color: badge.color }]}>{badge.label}</Text>
                        </View>
                      </View>
                      <Text style={styles.zoneDescription}>{zone.description}</Text>
                      <View style={styles.zonePressureRow}>
                        <Text style={styles.zonePressureLabel}>Pressure</Text>
                        <Text style={[styles.zonePressureValue, { color: badge.color }]}>{zone.crowdPressure}</Text>
                      </View>
                      <View style={styles.zoneTrendBar}>
                        <View
                          style={[
                            styles.zoneTrendFill,
                            { width: `${Math.max(12, Math.floor(zone.percent * 100))}%`, backgroundColor: zone.accent },
                          ]}
                        />
                      </View>
                    </View>
                  );
                })}
            </View>

            {topCrowdAlertZones.length > 0 ? (
              <View style={styles.alertFrequencySection}>
                <Text style={styles.sectionHeaderTitle}>Alert frequency</Text>
                <Text style={styles.sectionHeaderSub}>Recent crowd alert volume by zone</Text>
                <View style={styles.alertFrequencyGrid}>
                  {topCrowdAlertZones.map((zone) => {
                    const zoneInfo = crowdZoneProgress.find((item) => String(item.id) === String(zone.zone));
                    return (
                      <View key={zone.zone} style={styles.alertFrequencyCard}>
                        <Text style={styles.zoneCardLabel}>{zoneInfo?.label || `Zone ${zone.zone}`}</Text>
                        <Text style={styles.alertFrequencyCount}>{zone.count} alerts</Text>
                        <View style={styles.sparklineRow}>
                          {zone.sparkline.map((value, index) => (
                            <View
                              key={index}
                              style={[
                                styles.sparkBar,
                                {
                                  height: 16 + value * 4,
                                  backgroundColor: zoneInfo?.accent || colors.primary,
                                },
                              ]}
                            />
                          ))}
                        </View>
                      </View>
                    );
                  })}
                </View>
              </View>
            ) : null}

            <View style={styles.crowdProgressGroup}>
              {crowdZoneProgress.map((zone) => (
                <View key={zone.id} style={styles.crowdProgressRow}>
                  <View style={styles.crowdProgressRowHeader}>
                    <Text style={styles.crowdProgressLabel}>{zone.label}</Text>
                    <View style={styles.crowdMetaRow}>
                      <Text style={styles.crowdProgressMeta}>{zone.count} / {zone.threshold || "—"}</Text>
                      <View style={[styles.statusBadge, { backgroundColor: zone.status === "crowded" ? "rgba(239,68,68,0.16)" : zone.status === "elevated" ? "rgba(245,158,11,0.16)" : "rgba(34,197,94,0.16)", marginLeft: spacing.sm }]}>
                        <Text style={[styles.statusBadgeText, { color: zone.status === "crowded" ? colors.danger : zone.status === "elevated" ? colors.warning : colors.success }]}>
                          {zone.status?.toUpperCase() || "NORMAL"}
                        </Text>
                      </View>
                    </View>
                  </View>
                  <View style={styles.crowdProgressBarTrack}>
                    <View
                      style={[
                        styles.crowdProgressBarFill,
                        {
                          width: `${Math.floor(zone.percent * 100)}%`,
                          backgroundColor: zone.accent,
                          shadowColor: zone.accent,
                          shadowOpacity: 0.34,
                          shadowRadius: 6,
                          shadowOffset: { width: 0, height: 4 },
                          elevation: 2,
                        },
                      ]}
                    />
                  </View>
                </View>
              ))}
            </View>

            {crowdChart ? (
              <View style={styles.chartCard}>
                <View style={styles.crowdChartHeader}>
                  <View>
                    <Text style={styles.chartTitle}>Zone crowd levels</Text>
                    <Text style={styles.chartSubtitle}>Live crowd counts vs safe threshold for each zone.</Text>
                  </View>
                  <View style={styles.chartBadge}>
                    <Text style={styles.chartBadgeLabel}>Peak risk</Text>
                    <Text style={styles.chartBadgeValue}>{crowdZoneSummary.topZone?.zone_name || "—"}</Text>
                  </View>
                </View>
                <BarChart
                  data={crowdChart}
                  width={chartWidth}
                  height={layout.chartHeight}
                  chartConfig={chartConfig}
                  style={styles.chart}
                  fromZero
                  showBarTops
                  withInnerLines={false}
                  yAxisLabel=""
                  yAxisSuffix=""
                />
                <View style={styles.thresholdBanner}>
                  <Text style={styles.thresholdLabel}>Threshold line</Text>
                  <Text style={styles.thresholdValue}>Target: across all zones</Text>
                </View>
                <View style={styles.chartLegendRow}>
                  <View style={styles.chartLegendItem}>
                    <View style={[styles.chartLegendDot, { backgroundColor: "rgba(56,189,248,0.95)" }]} />
                    <Text style={styles.chartLegendText}>Current load</Text>
                  </View>
                  <View style={styles.chartLegendItem}>
                    <View style={[styles.chartLegendDot, { backgroundColor: "rgba(245,158,11,0.85)" }]} />
                    <Text style={styles.chartLegendText}>Threshold</Text>
                  </View>
                </View>
                {crowdAlerts.length > 0 ? (
                  <View style={styles.alertsCard}>
                    <Text style={styles.breakdownTitle}>Recent crowd alerts</Text>
                    {crowdAlerts.slice(0, 4).map((alert) => {
                      const alertTime = formatUpdatedAt(alert.timestamp || alert.ts || alert.ts_utc);
                      return (
                        <View key={`${alert.zone}-${alert.timestamp || alert.ts || alert.ts_utc}`} style={styles.alertRow}>
                          <View style={styles.alertDot} />
                          <View style={styles.alertMeta}>
                            <Text style={styles.alertLabel}>{alert.zone ? `Zone ${alert.zone}` : "Zone —"}</Text>
                            <Text style={styles.alertSubtitle}>{`${alert.person_count ?? alert.count ?? "?"} / ${alert.threshold ?? alert.limit ?? "?"} people`}</Text>
                          </View>
                          <Text style={styles.alertTime}>{alertTime}</Text>
                        </View>
                      );
                    })}
                  </View>
                ) : null}
              </View>
            ) : (
              <Text style={styles.emptyText}>Crowd analytics updates as zones report live counts.</Text>
            )}
          </>
        )}

        {activeTab === "response" && (
          <>
            {responseSummary ? (
              <View style={styles.cardGrid}>
                <View style={styles.statCard}>
                  <Text style={styles.statValue}>{responseSummary.total_responses ?? 0}</Text>
                  <Text style={styles.statLabel}>Total Responses</Text>
                </View>
                <View style={styles.statCard}>
                  <Text style={styles.statValue}>{responseSummary.avg_response_time ?? 0}s</Text>
                  <Text style={styles.statLabel}>Avg Response</Text>
                </View>
                <View style={styles.statCard}>
                  <Text style={[styles.statValue, { color: responseRisk.color }]}>{responseRisk.label}</Text>
                  <Text style={styles.statLabel}>Response Health</Text>
                </View>
                <View style={styles.statCard}>
                  <Text style={styles.statValue}>{responseSummary.max_response_time ?? 0}s</Text>
                  <Text style={styles.statLabel}>Slowest Response</Text>
                </View>
              </View>
            ) : null}
            {lifeguardRows.length > 0 ? (
              <View style={styles.breakdownCard}>
                <Text style={styles.breakdownTitle}>Top Lifeguard Performance</Text>
                {lifeguardRows.slice(0, 4).map((row) => (
                  <View key={row.id} style={styles.breakdownRow}>
                    <Text style={styles.breakdownLabel}>{row.name || `Lifeguard ${row.id}`}</Text>
                    <Text style={styles.breakdownValue}>{row.avg}s avg</Text>
                  </View>
                ))}
              </View>
            ) : null}
            {responseSummary?.status_counts ? (
              <View style={styles.breakdownCard}>
                <Text style={styles.breakdownTitle}>Response Statuses</Text>
                {Object.entries(responseSummary.status_counts).map(([status, count]) => (
                  <View key={status} style={styles.breakdownRow}>
                    <Text style={styles.breakdownLabel}>{status.replace(/_/g, " ")}</Text>
                    <Text style={styles.breakdownValue}>{count}</Text>
                  </View>
                ))}
              </View>
            ) : null}
            {zoneRows.length > 0 ? (
              <View style={styles.breakdownCard}>
                <Text style={styles.breakdownTitle}>Response by Zone</Text>
                {zoneRows.slice(0, 4).map((row) => (
                  <View key={row.zone} style={styles.breakdownRow}>
                    <Text style={styles.breakdownLabel}>Zone {row.zone}</Text>
                    <Text style={styles.breakdownValue}>{row.avg}s avg</Text>
                  </View>
                ))}
              </View>
            ) : null}
            {responseChart ? (
              <BarChart
                data={responseChart}
                width={chartWidth}
                height={layout.chartHeight}
                chartConfig={chartConfig}
                style={styles.chart}
                yAxisLabel=""
                yAxisSuffix="s"
              />
            ) : (
              <Text style={styles.emptyText}>
                No lifeguard responses recorded yet. Use the respond action when alerts arrive.
              </Text>
            )}
          </>
        )}

        {!loading && activeTab === "overview" && !overviewCards.length && !error && (
          <Text style={styles.emptyText}>No overview data available yet.</Text>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  body: {
    flex: 1,
  },
  content: {
    padding: spacing.md,
    paddingBottom: spacing.xl,
  },
  pageHeader: {
    backgroundColor: colors.surface,
    padding: spacing.md,
    marginHorizontal: spacing.md,
    marginBottom: spacing.md,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.border,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
  pageTitle: {
    color: colors.primary,
    fontSize: 22,
    fontWeight: "900",
    letterSpacing: 0.3,
    marginBottom: spacing.xs,
  },
  pageSubtitle: {
    color: colors.textMuted,
    fontSize: 13,
    lineHeight: 20,
    fontWeight: "600",
    marginBottom: spacing.sm,
  },
  sectionHeader: {
    backgroundColor: "rgba(16,34,53,0.72)",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.24)",
    padding: spacing.md,
    marginBottom: spacing.lg,
    shadowColor: "#000",
    shadowOpacity: 0.12,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
  premiumBanner: {
    backgroundColor: "rgba(53,214,195,0.08)",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.24)",
    padding: spacing.md,
    marginBottom: spacing.lg,
  },
  premiumBadge: {
    alignSelf: "flex-start",
    color: colors.primary,
    fontWeight: "900",
    fontSize: 12,
    letterSpacing: 1,
    textTransform: "uppercase",
    marginBottom: spacing.xs,
  },
  premiumText: {
    color: colors.textMuted,
    lineHeight: 20,
  },
  sectionHeaderTitle: {
    color: colors.text,
    fontWeight: "900",
    fontSize: 20,
    letterSpacing: 0.2,
  },
  sectionHeaderSub: {
    color: colors.textMuted,
    marginTop: 6,
    lineHeight: 20,
    fontSize: 13.5,
    fontWeight: "600",
  },
  cardGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
  },
  statCard: {
    width: "48%",
    marginBottom: spacing.md,
    backgroundColor: "rgba(16,34,53,0.68)",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.24)",
    padding: spacing.md,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  statCardWide: {
    width: "100%",
    backgroundColor: "rgba(16,34,53,0.68)",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.24)",
    padding: spacing.md,
    marginBottom: spacing.md,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  sectionTitle: {
    color: colors.text,
    fontWeight: "700",
    fontSize: 16,
    marginBottom: spacing.sm,
  },
  breakdownCard: {
    backgroundColor: "rgba(16,34,53,0.68)",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.24)",
    padding: spacing.md,
    marginTop: spacing.md,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  breakdownTitle: {
    color: colors.text,
    fontWeight: "800",
    marginBottom: spacing.md,
    fontSize: 16,
    letterSpacing: 0.15,
  },
  breakdownRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: spacing.xs,
  },
  breakdownRowCard: {
    backgroundColor: "rgba(16,34,53,0.68)",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.24)",
    padding: spacing.md,
    marginBottom: spacing.md,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  breakdownLabel: {
    color: colors.text,
    fontWeight: "700",
    fontSize: 14,
  },
  breakdownValue: {
    color: colors.primary,
    fontWeight: "900",
    fontSize: 20,
    marginTop: spacing.xs,
  },
  breakdownMeta: {
    color: colors.textMuted,
    marginTop: 2,
    fontSize: 12,
  },
  crowdSummaryCard: {
    width: "100%",
    backgroundColor: "rgba(255,255,255,0.04)",
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.18)",
    padding: spacing.md,
    marginBottom: spacing.md,
    shadowColor: colors.primary,
    shadowOpacity: 0.12,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 3,
  },
  crowdSummaryLabel: {
    color: colors.textMuted,
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: spacing.xs,
  },
  crowdRiskValue: {
    color: colors.primary,
    fontSize: 28,
    fontWeight: "900",
    letterSpacing: 0.5,
  },
  summaryStats: {
    marginTop: spacing.md,
    flexDirection: "row",
    justifyContent: "space-between",
    flexWrap: "wrap",
  },
  summaryStatItem: {
    flex: 1,
    marginBottom: spacing.md,
    padding: spacing.sm,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.12)",
  },
  summaryStatLabel: {
    color: colors.textMuted,
    fontSize: 11,
    fontWeight: "700",
    marginBottom: spacing.xs,
  },
  summaryStatValue: {
    color: colors.text,
    fontSize: 18,
    fontWeight: "900",
  },
  crowdProgressGroup: {
    marginTop: spacing.sm,
    marginBottom: spacing.md,
  },
  crowdProgressRow: {
    marginBottom: spacing.sm,
  },
  crowdProgressRowHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: spacing.xs,
  },
  crowdProgressLabel: {
    color: colors.text,
    fontSize: 14,
    fontWeight: "700",
  },
  crowdProgressMeta: {
    color: colors.textMuted,
    fontSize: 12,
  },
  crowdProgressBarTrack: {
    height: 10,
    borderRadius: 10,
    backgroundColor: "rgba(226,232,240,0.08)",
    overflow: "hidden",
  },
  crowdProgressBarFill: {
    height: "100%",
    borderRadius: 10,
    backgroundColor: colors.primary,
  },
  chartCard: {
    backgroundColor: "rgba(255,255,255,0.04)",
    borderRadius: 18,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.18)",
  },
  chartTitle: {
    color: colors.text,
    fontSize: 16,
    fontWeight: "900",
    marginBottom: spacing.xs,
  },
  chartSubtitle: {
    color: colors.textMuted,
    fontSize: 12,
    marginBottom: spacing.sm,
  },
  crowdChartHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: spacing.md,
  },
  chartBadge: {
    backgroundColor: "rgba(245,158,11,0.14)",
    borderRadius: 22,
    borderWidth: 1,
    borderColor: "rgba(245,158,11,0.24)",
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
  },
  chartBadgeLabel: {
    color: colors.textMuted,
    fontSize: 10,
    fontWeight: "700",
    marginBottom: spacing.xs / 2,
    textTransform: "uppercase",
  },
  chartBadgeValue: {
    color: colors.warning,
    fontSize: 13,
    fontWeight: "900",
  },
  statValue: {
    color: colors.primary,
    fontSize: 32,
    fontWeight: "900",
    lineHeight: 38,
  },
  statLabel: {
    color: colors.textMuted,
    marginTop: spacing.sm,
    fontSize: 13,
    fontWeight: "600",
  },
  chart: {
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.24)",
    marginTop: spacing.md,
    overflow: "hidden",
    backgroundColor: "rgba(16,34,53,0.92)",
    paddingVertical: spacing.sm,
  },
  crowdSummaryHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: spacing.md,
  },
  riskPill: {
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
    borderRadius: 999,
    borderWidth: 1,
  },
  riskPillText: {
    fontSize: 12,
    fontWeight: "800",
  },
  trendRow: {
    marginTop: spacing.md,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  trendLabel: {
    color: colors.textMuted,
    fontSize: 13,
    fontWeight: "700",
  },
  trendValue: {
    fontSize: 14,
    fontWeight: "900",
  },
  crowdMetaRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  crowdSummaryMeta: {
    alignItems: "flex-end",
  },
  updatedAtText: {
    color: colors.textMuted,
    fontSize: 11,
    marginTop: spacing.xs,
  },
  topZonesGroup: {
    flexDirection: "row",
    justifyContent: "space-between",
    flexWrap: "wrap",
    marginBottom: spacing.md,
  },
  zoneCard: {
    width: "48%",
    backgroundColor: "rgba(255,255,255,0.05)",
    borderRadius: 18,
    borderWidth: 1,
    padding: spacing.md,
    marginBottom: spacing.sm,
    borderColor: "rgba(255,255,255,0.08)",
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 3 },
    elevation: 2,
  },
  zoneCardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: spacing.xs,
  },
  zoneCardLabel: {
    color: colors.text,
    fontWeight: "800",
    fontSize: 14,
    flex: 1,
    marginRight: spacing.sm,
  },
  zoneDescription: {
    color: colors.textMuted,
    fontSize: 12,
    lineHeight: 18,
    marginBottom: spacing.sm,
  },
  zonePressureRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: spacing.sm,
  },
  zonePressureLabel: {
    color: colors.textMuted,
    fontSize: 12,
    fontWeight: "700",
  },
  zonePressureValue: {
    color: colors.text,
    fontWeight: "900",
    fontSize: 13,
  },
  zoneTrendBar: {
    height: 8,
    borderRadius: 8,
    backgroundColor: "rgba(226,232,240,0.08)",
    overflow: "hidden",
  },
  zoneTrendFill: {
    height: "100%",
    borderRadius: 8,
  },
  chartLegendRow: {
    marginTop: spacing.md,
    flexDirection: "row",
    justifyContent: "space-between",
  },
  chartLegendItem: {
    flexDirection: "row",
    alignItems: "center",
  },
  chartLegendDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: spacing.xs,
  },
  chartLegendText: {
    color: colors.textMuted,
    fontSize: 12,
  },
  alertsCard: {
    backgroundColor: "rgba(16,34,53,0.72)",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.18)",
    padding: spacing.md,
    marginTop: spacing.md,
  },
  alertRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: spacing.xs,
  },
  alertFrequencySection: {
    backgroundColor: "rgba(255,255,255,0.04)",
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.18)",
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  alertFrequencyGrid: {
    flexDirection: "row",
    justifyContent: "space-between",
    flexWrap: "wrap",
    marginTop: spacing.md,
  },
  alertFrequencyCard: {
    width: "48%",
    backgroundColor: "rgba(16,34,53,0.68)",
    borderRadius: 16,
    padding: spacing.sm,
    marginBottom: spacing.sm,
  },
  alertFrequencyCount: {
    color: colors.primary,
    fontWeight: "900",
    fontSize: 18,
    marginVertical: spacing.xs,
  },
  sparklineRow: {
    flexDirection: "row",
    alignItems: "flex-end",
    justifyContent: "space-between",
    height: 32,
  },
  sparkBar: {
    flex: 1,
    marginHorizontal: 2,
    borderRadius: 4,
  },
  alertDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.primary,
    marginRight: spacing.sm,
  },
  alertMeta: {
    flex: 1,
  },
  alertLabel: {
    color: colors.text,
    fontWeight: "700",
    fontSize: 13,
  },
  alertSubtitle: {
    color: colors.textMuted,
    fontSize: 11,
    marginTop: spacing.xs / 2,
  },
  alertTime: {
    color: colors.textMuted,
    fontSize: 11,
    marginLeft: spacing.sm,
    textAlign: "right",
  },
  statusBadge: {
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
    borderRadius: 999,
    borderWidth: 1,
  },
  statusBadgeText: {
    fontSize: 11,
    fontWeight: "800",
  },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.background,
    padding: spacing.lg,
  },
  emptyTitle: {
    color: colors.text,
    fontWeight: "700",
    fontSize: 18,
    textAlign: "center",
  },
  emptyText: {
    color: colors.textMuted,
    textAlign: "center",
    marginTop: spacing.lg,
  },
  error: {
    color: colors.danger,
    marginBottom: spacing.md,
  },
});
