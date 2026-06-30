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
  backgroundGradientFrom: colors.surface,
  backgroundGradientTo: colors.surfaceAlt,
  backgroundGradientFromOpacity: 0.94,
  backgroundGradientToOpacity: 0.96,
  decimalPlaces: 0,
  color: (opacity = 1) => `rgba(45, 212, 191, ${opacity})`,
  labelColor: (opacity = 1) => `rgba(226, 232, 240, ${opacity * 0.9})`,
  propsForDots: {
    r: "4",
    strokeWidth: "1",
    stroke: colors.primary,
  },
  propsForBackgroundLines: {
    stroke: "rgba(148,163,184,0.14)",
    strokeDasharray: "4",
  },
  fillShadowGradient: colors.primary,
  fillShadowGradientOpacity: 0.18,
  useShadowColorFromDataset: false,
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

export default function AnalyticsScreen() {
  const { baseUrl, api, connected, assignedZones } = useApiContext();
  const [activeTab, setActiveTab] = useState("overview");
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [crowdStatus, setCrowdStatus] = useState(null);
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
          const raw = await api.crowdStatus();
          const zones = filterCrowdZones(normalizeCrowdZones(raw), assignedZones);
          setCrowdStatus({
            ...raw,
            zones,
            crowded_zones_count: zones.filter((z) => z.exceeded || z.status === "crowded").length,
          });
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
    return {
      totalPeople,
      crowdedCount,
      topZone: zones
        .slice()
        .sort((a, b) => Number(b.person_count ?? b.count ?? 0) - Number(a.person_count ?? a.count ?? 0))[0],
    };
  }, [crowdZones]);

  const crowdZoneProgress = useMemo(() => {
    return crowdZones.map((zone, index) => {
      const count = Number(zone.person_count ?? zone.count ?? 0);
      const threshold = Number(zone.threshold ?? zone.limit ?? 0);
      const percent = threshold > 0 ? Math.min(count / threshold, 1) : 0;
      const colorIndex = index % ZONE_COLORS.length;
      return {
        id: zone.zone ?? zone.id,
        label: zone.zone_name || zone.name || `Zone ${zone.zone ?? zone.id}`,
        count,
        threshold,
        percent,
        status: zone.status || (threshold > 0 && count >= threshold ? "crowded" : "normal"),
        accent: ZONE_COLORS[colorIndex] || colors.primary,
      };
    });
  }, [crowdZones]);

  const crowdChart = useMemo(() => {
    if (!crowdZones.length) return null;
    const sorted = crowdZones
      .slice()
      .sort((a, b) => Number(b.person_count ?? b.count ?? 0) - Number(a.person_count ?? a.count ?? 0));
    return {
      labels: sorted.map((z) => z.zone_name || z.name || `Zone ${z.zone || z.id}`),
      datasets: [
        {
          data: sorted.map((z) => Number(z.person_count ?? z.count ?? 0)),
          color: (opacity = 1) => `rgba(56, 189, 248, ${opacity})`,
        },
      ],
    };
  }, [crowdZones]);

  const responseRows = useMemo(() => {
    return responseTimes?.recent || responseTimes?.items || responseTimes?.responses || [];
  }, [responseTimes]);

  const responseChart = useMemo(() => {
    if (!responseRows.length) return null;
    return {
      labels: responseRows.slice(0, 6).map((r) => `Z${r.zone}`),
      datasets: [{ data: responseRows.slice(0, 6).map((r) => Number(r.response_time_seconds ?? r.seconds ?? 0)) }],
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
                <View>
                  <Text style={styles.crowdSummaryLabel}>Live Crowd Risk</Text>
                  <Text style={styles.crowdRiskValue}>
                    {crowdStatus.overall_safety ? crowdStatus.overall_safety.toUpperCase() : "UNKNOWN"}
                  </Text>
                </View>
                <View style={styles.summaryStats}>
                  <View style={styles.summaryStatItem}>
                    <Text style={styles.summaryStatLabel}>Crowded Zones</Text>
                    <Text style={styles.summaryStatValue}>{crowdZoneSummary.crowdedCount ?? 0}</Text>
                  </View>
                  <View style={styles.summaryStatItem}>
                    <Text style={styles.summaryStatLabel}>People Total</Text>
                    <Text style={styles.summaryStatValue}>{crowdZoneSummary.totalPeople}</Text>
                  </View>
                </View>
              </View>
            ) : null}

            <View style={styles.crowdProgressGroup}>
              {crowdZoneProgress.map((zone) => (
                <View key={zone.id} style={styles.crowdProgressRow}>
                  <View style={styles.crowdProgressRowHeader}>
                    <Text style={styles.crowdProgressLabel}>{zone.label}</Text>
                    <Text style={styles.crowdProgressMeta}>
                      {zone.count} / {zone.threshold || "—"}
                    </Text>
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
                <Text style={styles.chartTitle}>Zone crowd levels</Text>
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
                  <Text style={styles.statValue}>{responseSummary.min_response_time ?? 0}s</Text>
                  <Text style={styles.statLabel}>Fastest Response</Text>
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
    gap: spacing.md,
  },
  statCard: {
    width: "48%",
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
    gap: spacing.md,
  },
  summaryStatItem: {
    flex: 1,
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
    gap: spacing.sm,
  },
  crowdProgressRow: {
    gap: spacing.xs,
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
    fontSize: 15,
    fontWeight: "800",
    marginBottom: spacing.sm,
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
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.22)",
    marginTop: spacing.md,
    overflow: "hidden",
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
