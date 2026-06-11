import React, { useCallback, useMemo, useState } from "react";
import { FlatList, RefreshControl, StyleSheet, View } from "react-native";
import { ActivityIndicator, Text } from "react-native-paper";
import { useNavigation } from "@react-navigation/native";
import ConnectionBanner from "../components/ConnectionBanner";
import ZoneCard from "../components/ZoneCard";
import { useApiContext } from "../context/ApiContext";
import { usePollApi } from "../hooks/usePollApi";
import { POLL_DETECTIONS_MS, POLL_ZONES_MS } from "../shared/constants";
import { colors, spacing } from "../theme";
import { filterZones } from "../utils/zoneFilter";

export default function DashboardScreen() {
  const navigation = useNavigation();
  const { baseUrl, api, connected, assignedZones, lifeguard, clearConnection, setConnected } = useApiContext();
  const [detectionMap, setDetectionMap] = useState({});
  const [refreshing, setRefreshing] = useState(false);

  const zonesPoll = usePollApi(
    () => api.zones(),
    POLL_ZONES_MS,
    Boolean(baseUrl),
    { items: [] },
    "zones"
  );

  const allZones = zonesPoll.data?.items || [];
  const zones = useMemo(() => filterZones(allZones, assignedZones), [allZones, assignedZones]);

  const loadDetections = useCallback(async () => {
    if (!zones.length) return;
    const entries = await Promise.all(
      zones.map(async (zone) => {
        try {
          const result = await api.detections(zone.id);
          return [zone.id, result?.count ?? 0];
        } catch {
          return [zone.id, 0];
        }
      })
    );
    setDetectionMap(Object.fromEntries(entries));
  }, [api, zones]);

  React.useEffect(() => {
    if (!baseUrl || !zones.length) return undefined;
    loadDetections();
    const timer = setInterval(loadDetections, POLL_DETECTIONS_MS);
    return () => clearInterval(timer);
  }, [baseUrl, zones, loadDetections]);

  React.useEffect(() => {
    if (zonesPoll.error) {
      clearConnection();
    } else if (zonesPoll.data) {
      setConnected(true);
    }
  }, [zonesPoll.data, zonesPoll.error, clearConnection, setConnected]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await zonesPoll.refresh();
      await loadDetections();
    } finally {
      setRefreshing(false);
    }
  }, [zonesPoll, loadDetections]);

  const content = useMemo(() => {
    if (!baseUrl) {
      return (
        <View style={styles.centered}>
          <Text style={styles.emptyTitle}>No server configured</Text>
          <Text style={styles.emptyText}>Open Settings and enter your laptop URL.</Text>
        </View>
      );
    }

    if (zonesPoll.loading && zones.length === 0) {
      return (
        <View style={styles.centered}>
          <ActivityIndicator color={colors.primary} />
        </View>
      );
    }

    if (zones.length === 0) {
      return (
        <View style={styles.centered}>
          <Text style={styles.emptyTitle}>
            {allZones.length > 0 && assignedZones?.length > 0 ? "No assigned zones" : "No zones found"}
          </Text>
          <Text style={styles.emptyText}>
            {allZones.length > 0 && assignedZones?.length > 0
              ? "Ask your admin to assign zones in the web Lifeguards tab."
              : "Upload videos to the backend and reload zones."}
          </Text>
        </View>
      );
    }

    return (
      <FlatList
        data={zones}
        keyExtractor={(item) => String(item.id)}
        contentContainerStyle={styles.list}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.primary} />}
        renderItem={({ item, index }) => (
          <ZoneCard
            zone={item}
            index={index}
            personCount={detectionMap[item.id] ?? 0}
            frameUrl={api.frameUrl(item.id, 640)}
            onPress={() => navigation.navigate("ZoneDetail", { zone: item })}
          />
        )}
      />
    );
  }, [allZones, assignedZones, api, baseUrl, detectionMap, navigation, onRefresh, refreshing, zones, zonesPoll.loading]);

  return (
    <View style={styles.container}>
      <ConnectionBanner visible={Boolean(baseUrl) && !connected} />
      {lifeguard && assignedZones?.length > 0 ? (
        <View style={styles.scopeBanner}>
          <Text style={styles.scopeText}>
            Showing {zones.length} assigned zone{zones.length === 1 ? "" : "s"} for {lifeguard.name}
          </Text>
        </View>
      ) : null}
      {content}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  scopeBanner: {
    backgroundColor: colors.surface,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  scopeText: {
    color: colors.textMuted,
    fontSize: 12,
    fontWeight: "600",
  },
  list: {
    padding: spacing.md,
    paddingBottom: spacing.xl,
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
  },
  emptyText: {
    color: colors.textMuted,
    textAlign: "center",
    lineHeight: 20,
  },
});
