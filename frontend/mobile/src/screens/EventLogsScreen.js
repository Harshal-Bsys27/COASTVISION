import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, FlatList, Pressable, StyleSheet, View } from "react-native";
import { ActivityIndicator, Snackbar, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import ActivityEventCard from "../components/ActivityEventCard";
import ConnectionBanner from "../components/ConnectionBanner";
import PollErrorBanner from "../components/PollErrorBanner";
import TabChipRow from "../components/TabChipRow";
import { useApiContext } from "../context/ApiContext";
import { usePollApi } from "../hooks/usePollApi";
import { useAlertNotifications } from "../hooks/useAlertNotifications";
import { DEFAULT_ALERT_LIMIT, LOG_FILTER_TABS, POLL_ALERTS_MS } from "../shared/constants";
import {
  crowdAlertsToEvents,
  crowdStatusChangesToEvents,
  drowningAlertsToEvents,
  filterActivityEvents,
  groupRawDetections,
  mergeActivityEvents,
  responsesToEvents,
} from "../utils/activityEvents";
import { colors, spacing } from "../theme";
import { filterAlerts, filterByZoneField } from "../utils/zoneFilter";
import { logInfo } from "../utils/logger";

function normalizeCrowdZonesFromStatus(crowdStatus) {
  const raw = crowdStatus?.zones ?? crowdStatus?.items;
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  return Object.entries(raw).map(([zoneId, zone]) => ({
    zone: zoneId,
    id: zoneId,
    ...zone,
  }));
}

export default function EventLogsScreen() {
  const { baseUrl, api, connected, lifeguard, assignedZones, clearConnection, setConnected } = useApiContext();
  const [activeFilter, setActiveFilter] = useState("all");
  const [showDeveloperLogs, setShowDeveloperLogs] = useState(false);
  const [statusChangeEvents, setStatusChangeEvents] = useState([]);
  const [respondingId, setRespondingId] = useState(null);
  const [respondedIds, setRespondedIds] = useState(() => new Set());
  const [snackbar, setSnackbar] = useState({ visible: false, message: "" });
  const previousCrowdRef = useRef({});

  const alertsPoll = usePollApi(
    () =>
      lifeguard?.id
        ? api.lifeguardAlerts(lifeguard.id, DEFAULT_ALERT_LIMIT).then((data) => ({
            items: data.alerts || [],
          }))
        : api.alerts(DEFAULT_ALERT_LIMIT),
    POLL_ALERTS_MS,
    Boolean(baseUrl),
    { items: [] },
    "alerts"
  );

  const crowdAlertsPoll = usePollApi(
    () => api.crowdAlerts(80),
    POLL_ALERTS_MS,
    Boolean(baseUrl),
    { alerts: [] },
    "crowd-alerts"
  );

  const responsesPoll = usePollApi(
    () => api.responseTimes(50),
    POLL_ALERTS_MS,
    Boolean(baseUrl),
    { recent: [] },
    "response-times"
  );

  const submitResponse = useCallback(
    async (event, status) => {
      if (!lifeguard?.id || respondingId) return;
      setRespondingId(event.id);
      try {
        const result = await api.lifeguardRespond(lifeguard.id, event.alertId, event.zone, status);
        setRespondedIds((prev) => new Set(prev).add(event.id));
        const seconds = result?.response_time_seconds;
        const label = status === "en_route" ? "En route" : "Acknowledged";
        const message = seconds != null
          ? `${label} (${Number(seconds).toFixed(1)}s)`
          : `${label} response recorded`;
        setSnackbar({ visible: true, message });
        logInfo("Lifeguard responded", { zone: event.zone, alertId: event.alertId, status, seconds });
        responsesPoll.refresh();
      } catch (err) {
        setSnackbar({
          visible: true,
          message: err?.message || "Could not record response",
        });
      } finally {
        setRespondingId(null);
      }
    },
    [api, lifeguard?.id, respondingId, responsesPoll]
  );

  const handleRespond = useCallback(
    (event) => {
      if (!lifeguard?.id || respondingId) return;

      Alert.alert(
        "Confirm response",
        `Choose how you are responding to zone ${event.zone}`,
        [
          {
            text: "Acknowledge",
            onPress: () => submitResponse(event, "acknowledged"),
          },
          {
            text: "En route",
            onPress: () => submitResponse(event, "en_route"),
          },
          {
            text: "Cancel",
            style: "cancel",
          },
        ],
        { cancelable: true }
      );
    },
    [lifeguard?.id, respondingId, submitResponse]
  );

  const crowdStatusPoll = usePollApi(
    () => api.crowdStatus(),
    POLL_ALERTS_MS,
    Boolean(baseUrl),
    null,
    "crowd-status"
  );

  const alerts = useMemo(
    () => filterAlerts(alertsPoll.data?.items || [], assignedZones),
    [alertsPoll.data, assignedZones]
  );
  useAlertNotifications(alerts, Boolean(baseUrl) && connected);

  useEffect(() => {
    if (alertsPoll.error) {
      clearConnection();
    } else if (alertsPoll.data) {
      setConnected(true);
    }
  }, [alertsPoll.data, alertsPoll.error, clearConnection, setConnected]);

  useEffect(() => {
    const currentZones = filterByZoneField(
      normalizeCrowdZonesFromStatus(crowdStatusPoll.data),
      assignedZones,
      (zone) => zone.zone || zone.id
    );
    if (!currentZones.length) return;

    const previous = previousCrowdRef.current;
    if (Object.keys(previous).length > 0) {
      const changes = crowdStatusChangesToEvents(previous, currentZones);
      if (changes.length) {
        setStatusChangeEvents((current) => mergeActivityEvents(changes, current).slice(0, 40));
      }
    }

    previousCrowdRef.current = Object.fromEntries(
      currentZones.map((zone) => [String(zone.zone || zone.id), zone])
    );
  }, [assignedZones, crowdStatusPoll.data]);

  const activityEvents = useMemo(() => {
    const crowdAlerts = filterByZoneField(
      crowdAlertsPoll.data?.alerts || [],
      assignedZones,
      (alert) => alert.zone
    );
    const merged = mergeActivityEvents(
      crowdAlertsToEvents(crowdAlerts),
      responsesToEvents(responsesPoll.data?.recent || []),
      drowningAlertsToEvents(alerts),
      statusChangeEvents
    );
    return filterByZoneField(merged, assignedZones, (event) => event.zone);
  }, [alerts, assignedZones, crowdAlertsPoll.data, responsesPoll.data, statusChangeEvents]);

  const developerEvents = useMemo(() => groupRawDetections(alerts), [alerts]);

  const respondedIdsFromApi = useMemo(() => {
    const rows = responsesPoll.data?.recent || [];
    const ids = new Set();
    activityEvents.forEach((event) => {
      if (!event.respondable) return;
      const eventZone = String(event.zone);
      const eventTime = event.timestamp || 0;
      const matched = rows.some((row) => {
        if (String(row.zone) !== eventZone) return false;
        const rowTime = new Date(row.responded_at || row.timestamp || 0).getTime();
        if (!rowTime || !eventTime) return true;
        return Math.abs(rowTime - eventTime) < 15 * 60 * 1000;
      });
      if (matched) ids.add(event.id);
    });
    return ids;
  }, [activityEvents, responsesPoll.data]);

  const filteredEvents = useMemo(
    () => filterActivityEvents(activityEvents, activeFilter),
    [activityEvents, activeFilter]
  );

  const loading =
    (alertsPoll.loading && !alerts.length) ||
    (crowdAlertsPoll.loading && !(crowdAlertsPoll.data?.alerts || []).length);

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
        <Text style={styles.pageTitle}>Alert Response Center</Text>
        <Text style={styles.pageSubtitle}>Monitor live incidents, triage alerts, and confirm acknowledgments.</Text>
      </View>
      <PollErrorBanner visible={Boolean(crowdAlertsPoll.error)} message={crowdAlertsPoll.error?.message} />
      <PollErrorBanner visible={Boolean(responsesPoll.error)} message={responsesPoll.error?.message} />
      <TabChipRow tabs={LOG_FILTER_TABS} activeId={activeFilter} onSelect={setActiveFilter} />

      <View style={styles.headerCard}>
        <Text style={styles.headerTitle}>Activity Timeline</Text>
        <Text style={styles.headerSub}>Operational incidents, crowd warnings, and lifeguard responses</Text>
      </View>

      <View style={styles.infoCard}>
        <Text style={styles.infoTitle}>Live response readiness</Text>
        <Text style={styles.infoText}>
          This stream shows current crowd warnings, fire drills, and alert acknowledgments across your assigned zones.
        </Text>
      </View>

      {loading && filteredEvents.length === 0 ? (
        <View style={styles.centered}>
          <ActivityIndicator color={colors.primary} />
        </View>
      ) : (
        <FlatList
          data={filteredEvents}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.list}
          ListEmptyComponent={
            <View style={styles.centered}>
              <Text style={styles.emptyTitle}>No activity events yet</Text>
              <Text style={styles.emptyText}>
                Significant crowd, alert, and response events will appear here.
              </Text>
            </View>
          }
          ListFooterComponent={
            <View style={styles.developerSection}>
              <Pressable style={styles.developerHeader} onPress={() => setShowDeveloperLogs((v) => !v)}>
                <MaterialCommunityIcons
                  name={showDeveloperLogs ? "chevron-up" : "chevron-down"}
                  size={20}
                  color={colors.textMuted}
                />
                <Text style={styles.developerTitle}>Developer Logs</Text>
                <Text style={styles.developerCount}>{developerEvents.length}</Text>
              </Pressable>

              {showDeveloperLogs ? (
                developerEvents.length ? (
                  developerEvents.map((event) => <ActivityEventCard key={event.id} event={event} />)
                ) : (
                  <Text style={styles.emptyText}>No grouped raw detections yet.</Text>
                )
              ) : (
                <Text style={styles.developerHint}>Hidden raw YOLO detections grouped by zone and type.</Text>
              )}
            </View>
          }
          renderItem={({ item }) => (
            <ActivityEventCard
              event={item}
              onRespond={item.respondable ? handleRespond : undefined}
              responding={respondingId === item.id}
              responded={respondedIds.has(item.id) || respondedIdsFromApi.has(item.id)}
            />
          )}
        />
      )}
      <Snackbar
        visible={snackbar.visible}
        onDismiss={() => setSnackbar((s) => ({ ...s, visible: false }))}
        duration={3000}
      >
        {snackbar.message}
      </Snackbar>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  list: {
    padding: spacing.md,
    paddingBottom: spacing.xl,
  },
  pageHeader: {
    backgroundColor: colors.surface,
    padding: spacing.md,
    marginHorizontal: spacing.md,
    marginTop: spacing.md,
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
  headerCard: {
    marginHorizontal: spacing.md,
    marginTop: 0,
    marginBottom: spacing.md,
    backgroundColor: "rgba(16,34,53,0.72)",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.24)",
    padding: spacing.md,
    shadowColor: "#000",
    shadowOpacity: 0.12,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
  infoCard: {
    marginHorizontal: spacing.md,
    marginBottom: spacing.md,
    backgroundColor: "rgba(53,214,195,0.08)",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.24)",
    padding: spacing.md,
  },
  infoTitle: {
    color: colors.primary,
    fontWeight: "900",
    fontSize: 14,
    marginBottom: spacing.xs,
  },
  infoText: {
    color: colors.textMuted,
    lineHeight: 20,
  },
  headerTitle: {
    color: colors.text,
    fontSize: 20,
    fontWeight: "900",
    letterSpacing: 0.2,
  },
  headerSub: {
    color: colors.textMuted,
    marginTop: 6,
    lineHeight: 20,
    fontSize: 13.5,
    fontWeight: "600",
  },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.lg,
  },
  emptyTitle: {
    color: colors.text,
    fontWeight: "700",
    fontSize: 18,
    marginBottom: spacing.sm,
    textAlign: "center",
  },
  emptyText: {
    color: colors.textMuted,
    textAlign: "center",
    lineHeight: 20,
  },
  developerSection: {
    marginTop: spacing.lg,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: "rgba(53,214,195,0.18)",
    backgroundColor: "rgba(16,34,53,0.42)",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.14)",
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.md,
  },
  developerHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    marginBottom: spacing.md,
    paddingTop: spacing.sm,
  },
  developerTitle: {
    color: colors.text,
    fontWeight: "800",
    flex: 1,
    fontSize: 15,
  },
  developerCount: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: "700",
    backgroundColor: "rgba(53,214,195,0.22)",
    borderRadius: 12,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
  },
  developerHint: {
    color: colors.textMuted,
    fontSize: 13,
    lineHeight: 19,
    fontWeight: "600",
  },
});
