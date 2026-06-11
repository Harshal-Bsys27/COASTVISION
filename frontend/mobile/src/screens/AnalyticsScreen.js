import React, { useCallback, useMemo, useState } from "react";
import { Dimensions, RefreshControl, ScrollView, StyleSheet, View } from "react-native";
import { ActivityIndicator, Text } from "react-native-paper";
import { BarChart, LineChart } from "react-native-chart-kit";
import ConnectionBanner from "../components/ConnectionBanner";
import TabChipRow from "../components/TabChipRow";
import { useApiContext } from "../context/ApiContext";
import { ANALYTICS_TABS } from "../shared/constants";
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
  decimalPlaces: 0,
  color: (opacity = 1) => `rgba(45, 212, 191, ${opacity})`,
  labelColor: () => colors.textMuted,
  propsForDots: {
    r: "3",
    strokeWidth: "1",
    stroke: colors.primary,
  },
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

  const crowdChart = useMemo(() => {
    if (!crowdZones.length) return null;
    return {
      labels: crowdZones.map((z) => z.zone_name || `Z${z.zone || z.id}`),
      datasets: [{ data: crowdZones.map((z) => Number(z.person_count ?? z.count ?? 0)) }],
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
      <TabChipRow tabs={ANALYTICS_TABS} activeId={activeTab} onSelect={setActiveTab} />

      <ScrollView
        style={styles.body}
        contentContainerStyle={styles.content}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.primary} />}
      >
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
              <View style={styles.statCardWide}>
                <Text style={styles.statValue}>{crowdStatus.crowded_zones_count ?? 0}</Text>
                <Text style={styles.statLabel}>
                  Crowded zones · Overall: {crowdStatus.overall_safety || "unknown"}
                </Text>
              </View>
            ) : null}
            {crowdZones.map((zone) => (
              <View key={zone.zone || zone.id} style={styles.breakdownRowCard}>
                <Text style={styles.breakdownLabel}>{zone.zone_name || `Zone ${zone.zone || zone.id}`}</Text>
                <Text style={styles.breakdownValue}>
                  {zone.person_count ?? zone.count ?? 0} / {zone.threshold ?? "—"} people
                </Text>
                <Text style={styles.breakdownMeta}>
                  {zone.status || "normal"} · {zone.safety_percentage ?? "—"}% safe
                </Text>
              </View>
            ))}
            {crowdChart ? (
              <BarChart
                data={crowdChart}
                width={chartWidth}
                height={layout.chartHeight}
                chartConfig={chartConfig}
                style={styles.chart}
                yAxisLabel=""
                yAxisSuffix=""
              />
            ) : null}
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
  cardGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.sm,
  },
  statCard: {
    width: "48%",
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: spacing.md,
  },
  statCardWide: {
    width: "100%",
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: spacing.md,
    marginBottom: spacing.sm,
  },
  sectionTitle: {
    color: colors.text,
    fontWeight: "700",
    fontSize: 16,
    marginBottom: spacing.sm,
  },
  breakdownCard: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: spacing.md,
    marginTop: spacing.md,
  },
  breakdownTitle: {
    color: colors.text,
    fontWeight: "700",
    marginBottom: spacing.sm,
  },
  breakdownRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: spacing.xs,
  },
  breakdownRowCard: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: spacing.md,
    marginBottom: spacing.sm,
  },
  breakdownLabel: {
    color: colors.text,
    fontWeight: "600",
  },
  breakdownValue: {
    color: colors.primary,
    fontWeight: "800",
    fontSize: 18,
    marginTop: spacing.xs,
  },
  breakdownMeta: {
    color: colors.textMuted,
    marginTop: 2,
    fontSize: 12,
  },
  statValue: {
    color: colors.primary,
    fontSize: 28,
    fontWeight: "800",
  },
  statLabel: {
    color: colors.textMuted,
    marginTop: spacing.xs,
  },
  chart: {
    borderRadius: 12,
    marginTop: spacing.sm,
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
