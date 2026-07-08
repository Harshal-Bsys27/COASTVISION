import React, { useCallback, useMemo, useState } from "react";
import { FlatList, Image, RefreshControl, StyleSheet, View, useWindowDimensions, Alert } from "react-native";
import { ActivityIndicator, Text, Button } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useIsFocused, useNavigation } from "@react-navigation/native";
import ConnectionBanner from "../components/ConnectionBanner";
import ZoneCard from "../components/ZoneCard";
import { useApiContext } from "../context/ApiContext";
import { usePollApi } from "../hooks/usePollApi";
import { POLL_DETECTIONS_MS, POLL_HEALTH_MS, POLL_ZONES_MS } from "../shared/constants";
import { normalizeBaseUrl } from "../shared/api";
import { colors, spacing } from "../theme";
import { filterZones } from "../utils/zoneFilter";

function createFallbackAvatarUri(name = "LF") {
  const initials = String(name || "LF")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0] || "")
    .join("")
    .toUpperCase() || "LF";
  const bg = "#1f4d7a";
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240"><rect width="100%" height="100%" rx="120" fill="${bg}"/><circle cx="120" cy="90" r="36" fill="#8bd9ff"/><path d="M50 200c12-34 40-52 70-52s58 18 70 52" fill="#8bd9ff"/><text x="120" y="220" text-anchor="middle" font-family="Arial, sans-serif" font-size="72" font-weight="700" fill="#ffffff">${initials}</text></svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function resolveAvatarUri(profile, baseUrl) {
  const candidate = profile?.avatar_thumb_url || profile?.avatar_url || profile?.avatar_thumb || profile?.avatar;
  if (!candidate || typeof candidate !== "string") {
    return null;
  }

  if (candidate.startsWith("data:")) {
    return null;
  }

  const normalizedBase = normalizeBaseUrl(baseUrl);
  if (candidate.startsWith("/")) {
    return normalizedBase ? `${normalizedBase}${candidate}` : candidate;
  }

  if (/^https?:\/\//i.test(candidate)) {
    try {
      const parsed = new URL(candidate);
      const base = normalizedBase ? new URL(normalizedBase) : null;
      if (base && ["localhost", "127.0.0.1", "0.0.0.0", "::1"].includes(parsed.hostname)) {
        return `${base.origin}${parsed.pathname}${parsed.search}${parsed.hash}`;
      }
      if (base && parsed.origin === base.origin) {
        return `${base.origin}${parsed.pathname}${parsed.search}${parsed.hash}`;
      }
      return normalizedBase ? `${normalizedBase}${parsed.pathname}${parsed.search}${parsed.hash}` : candidate;
    } catch {
      return normalizedBase ? `${normalizedBase}${candidate}` : candidate;
    }
  }

  return normalizedBase ? `${normalizedBase}/${candidate.replace(/^\/+/, "")}` : candidate;
}

export default function DashboardScreen() {
  const navigation = useNavigation();
  const isFocused = useIsFocused();
  const { width } = useWindowDimensions();
  const { baseUrl, api, connected, assignedZones, lifeguard, clearConnection, setConnected, streamQuality, refreshLifeguard } = useApiContext();
  const [detectionMap, setDetectionMap] = useState({});
  const [refreshing, setRefreshing] = useState(false);
  const [sosActive, setSosActive] = useState(false);
  const [multiZoneView, setMultiZoneView] = useState(true);
  const [avatarLoadError, setAvatarLoadError] = useState(false);
  const fallbackAvatarUri = useMemo(() => createFallbackAvatarUri(lifeguard?.name || lifeguard?.id || "LF"), [lifeguard?.name, lifeguard?.id]);
  const isLargeViewport = width >= 920;
  const avatarUri = useMemo(
    () => resolveAvatarUri(lifeguard, baseUrl),
    [baseUrl, lifeguard?.avatar_url, lifeguard?.avatar, lifeguard?.avatar_thumb_url, lifeguard?.avatar_thumb, lifeguard?.name, lifeguard?.id]
  );
  const isRemoteAvatarUri = typeof avatarUri === "string" && /^(https?:\/\/)/i.test(avatarUri);
  const showAvatarImage = Boolean(avatarUri) && isRemoteAvatarUri && !avatarLoadError;
  const displayAvatarUri = avatarLoadError ? fallbackAvatarUri : avatarUri;

  const zonesPoll = usePollApi(
    () => api.zones(),
    POLL_ZONES_MS,
    Boolean(baseUrl),
    { items: [] },
    "zones"
  );

  const responseAnalyticsPoll = usePollApi(
    () => api.responseTimes(20),
    POLL_HEALTH_MS * 2,
    Boolean(baseUrl),
    null,
    "response-times"
  );

  const allZones = zonesPoll.data?.items || [];
  const zones = useMemo(() => filterZones(allZones, assignedZones), [allZones, assignedZones]);
  const placeholderZones = useMemo(() => {
    if (zones.length || !baseUrl || !zonesPoll.loading) return [];
    if (Array.isArray(assignedZones) && assignedZones.length > 0) {
      return assignedZones.slice(0, Math.min(2, assignedZones.length)).map((zoneId, index) => ({
        id: `placeholder-${zoneId || index}`,
        name: `Zone ${zoneId}`,
        placeholder: true,
      }));
    }
    return [{ id: "placeholder-1", name: "Loading zone", placeholder: true }];
  }, [assignedZones, baseUrl, zones.length, zonesPoll.loading]);
  const isSingleZone = zones.length <= 1;
  const canMultiColumn = zones.length > 1 && width >= 760;
  const cardColumns = multiZoneView && canMultiColumn ? 2 : 1;
  const assignedZoneLabel = zones.length === 1 ? zones[0]?.name || `Zone ${zones[0]?.id}` : zones.length > 1 ? `${zones.length} assigned zones` : "No assigned zones";
  const assignedZoneSummary = zones.length === 1 ? `You are currently assigned to ${assignedZoneLabel}.` : zones.length > 1 ? `You are currently assigned to ${zones.length} zones.` : "No assigned zones available yet.";
  const statusText = connected ? "Online" : "Offline";
  const statusColor = connected ? colors.success : colors.danger;
  const responseSummary = responseAnalyticsPoll.data?.overall || {};
  const responseBadge = responseSummary.total_responses
    ? `${responseSummary.avg_response_time}s avg · ${responseSummary.total_responses} responses`
    : "No response analytics yet";

  const zoneStreamHeight = useMemo(() => {
    const horizontalPadding = spacing.md * 2;
    const gutters = cardColumns > 1 ? spacing.sm : 0;
    const usableWidth = Math.max(320, width - horizontalPadding - gutters);
    const estimatedCardWidth = cardColumns > 1 ? usableWidth / 2 : Math.min(usableWidth, 1140);
    const computed = Math.round(estimatedCardWidth * 0.52);
    return Math.max(250, Math.min(420, computed));
  }, [width, cardColumns]);

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
    setAvatarLoadError(false);
  }, [avatarUri]);

  React.useEffect(() => {
    if (zonesPoll.error) {
      clearConnection();
    } else if (zonesPoll.data) {
      setConnected(true);
    }
  }, [zonesPoll.data, zonesPoll.error, clearConnection, setConnected]);

  React.useEffect(() => {
    if (!isFocused || !baseUrl || !lifeguard?.id) return;
    refreshLifeguard()
      .then(() => setAvatarLoadError(false))
      .catch(() => {});
  }, [isFocused, baseUrl, lifeguard?.id, refreshLifeguard]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await zonesPoll.refresh();
      await loadDetections();
    } finally {
      setRefreshing(false);
    }
  }, [zonesPoll, loadDetections]);

  const triggerEmergencySOS = useCallback(() => {
    Alert.alert(
      "🚨 EMERGENCY SOS",
      "Confirm emergency situation - all lifeguards will be alerted immediately",
      [
        {
          text: "Cancel",
          onPress: () => setSosActive(false),
          style: "cancel",
        },
        {
          text: "CONFIRM SOS",
          onPress: () => {
            setSosActive(true);
            Alert.alert(
              "✅ SOS Activated",
              `Emergency alert sent to all lifeguards at zone ${zones[0]?.name || ""}`,
              [{ text: "OK", onPress: () => setTimeout(() => setSosActive(false), 5000) }]
            );
          },
          style: "destructive",
        },
      ]
    );
  }, [zones]);

  const content = useMemo(() => {
    if (!baseUrl) {
      return (
        <View style={styles.centered}>
          <Text style={styles.emptyTitle}>No server configured</Text>
          <Text style={styles.emptyText}>Open Settings and enter your laptop URL.</Text>
        </View>
      );
    }

    if (zonesPoll.loading && zones.length === 0 && placeholderZones.length === 0) {
      return (
        <View style={styles.centered}>
          <ActivityIndicator color={colors.primary} />
        </View>
      );
    }

    if (zones.length === 0 && placeholderZones.length === 0) {
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

    const activeZones = zones.length ? zones : placeholderZones;

    return (
      <FlatList
        key={`zones-${cardColumns}`}
        data={activeZones}
        numColumns={cardColumns}
        keyExtractor={(item) => String(item.id)}
        contentContainerStyle={[styles.list, isSingleZone && styles.listSingle]}
        columnWrapperStyle={cardColumns > 1 ? styles.columnWrap : undefined}
        ListHeaderComponent={
          <View>
            <View style={styles.heroCard}>
              <View style={styles.heroTop}>
                <Text style={styles.heroTitle}>CoastVision Command</Text>
                <View style={styles.heroBadge}>
                  <MaterialCommunityIcons name="waves" size={14} color={colors.primary} />
                  <Text style={styles.heroBadgeText}>{zones.length} active zone{zones.length === 1 ? "" : "s"}</Text>
                </View>
              </View>
              <Text style={styles.heroSubtitle}>Real-time surveillance of your assigned coastal area</Text>
            </View>
            <View style={styles.heroActions}>
              <Button
                icon="alert-circle"
                mode="contained"
                buttonColor={sosActive ? "#dc2626" : "#ef4444"}
                textColor="#fff"
                onPress={triggerEmergencySOS}
                style={styles.sosButton}
                labelStyle={styles.sosButtonLabel}
              >
                {sosActive ? "🚨 SOS ACTIVE" : "🚨 EMERGENCY SOS"}
              </Button>
              {zones.length > 1 && canMultiColumn ? (
                <Button
                  icon={multiZoneView ? "view-grid" : "view-list"}
                  mode={multiZoneView ? "contained" : "outlined"}
                  buttonColor={multiZoneView ? "rgba(53,214,195,0.16)" : undefined}
                  textColor={multiZoneView ? colors.primary : colors.text}
                  onPress={() => setMultiZoneView((prev) => !prev)}
                  style={styles.layoutToggle}
                  labelStyle={styles.layoutToggleLabel}
                >
                  {multiZoneView ? "Multi-Zone" : "Single View"}
                </Button>
              ) : null}
            </View>
          </View>
        }
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.primary} />}
        renderItem={({ item, index }) => (
          <View style={[styles.zoneCell, isSingleZone && styles.zoneCellSingle]}>
            <ZoneCard
              zone={item}
              index={index}
              personCount={detectionMap[item.id] ?? 0}
              frameUrl={item.placeholder ? null : api.frameUrl(item.id, 640)}
              streamHeight={zoneStreamHeight}
              fps={streamQuality}
              onPress={() => !item.placeholder && navigation.navigate("ZoneDetail", { zone: item })}
            />
          </View>
        )}
      />
    );
  }, [allZones, api, assignedZones, baseUrl, cardColumns, detectionMap, isSingleZone, navigation, onRefresh, placeholderZones, refreshing, zoneStreamHeight, zones, zonesPoll.loading]);

  return (
    <View style={styles.container}>
      <ConnectionBanner visible={Boolean(baseUrl) && !connected} />
      <View style={styles.pageHeader}>
        <Text style={styles.pageTitle}>Lifeguard Command Deck</Text>
        <Text style={styles.pageSubtitle}>Live zone monitoring, emergency alerts, and operational controls.</Text>
      </View>
      {lifeguard ? (
        <View style={styles.scopeBanner}>
          {showAvatarImage ? (
            <View style={styles.avatarFrame}>
              <Image
                key={displayAvatarUri}
                source={{ uri: displayAvatarUri }}
                style={styles.avatar}
                onError={() => {
                  setAvatarLoadError(true);
                }}
              />
            </View>
          ) : (
            <View style={styles.avatarFallback}>
              <Text style={styles.avatarInitials}>{(lifeguard?.name || lifeguard?.id || "LF").split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0]).join("").toUpperCase() || "LF"}</Text>
            </View>
          )}
          <View style={styles.scopeTextWrap}>
            <Text style={styles.scopeTitle}>Welcome, Lifeguard {lifeguard.name}</Text>
            <Text style={styles.scopeText}>{assignedZoneSummary}</Text>
            <Text style={styles.responseSummary}>{responseBadge}</Text>
          </View>
          <View style={styles.statusBadge}>
            <Text style={[styles.statusBadgeText, { color: statusColor }]}>{statusText}</Text>
          </View>
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
    padding: spacing.md,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: colors.border,
    marginHorizontal: spacing.md,
    marginBottom: spacing.md,
    flexDirection: "row",
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 4 },
    elevation: 4,
  },
  scopeTextWrap: {
    flex: 1,
  },
  statusBadge: {
    backgroundColor: "rgba(53,214,195,0.12)",
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.24)",
  },
  statusBadgeText: {
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0.6,
  },
  scopeTitle: {
    color: colors.text,
    fontSize: 15,
    fontWeight: "800",
    marginBottom: 2,
  },
  scopeText: {
    color: colors.textMuted,
    fontSize: 14,
    fontWeight: "700",
    marginTop: spacing.xs,
    lineHeight: 22,
  },
  analyticsCard: {
    marginTop: spacing.sm,
    padding: spacing.sm,
    backgroundColor: "rgba(53,214,195,0.08)",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.28)",
  },
  analyticsTitle: {
    color: colors.textMuted,
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.8,
    marginBottom: spacing.xs,
  },
  analyticsValue: {
    color: colors.text,
    fontSize: 16,
    fontWeight: "900",
    lineHeight: 24,
  },
  responseSummary: {
    color: colors.textMuted,
    fontSize: 12,
    fontWeight: "600",
    marginTop: spacing.xs,
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
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    marginRight: spacing.sm,
    borderWidth: 2,
    borderColor: "rgba(53,214,195,0.25)",
    backgroundColor: "rgba(16,34,53,0.9)",
  },
  avatarFallback: {
    width: 80,
    height: 80,
    borderRadius: 40,
    marginRight: spacing.sm,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(53,214,195,0.14)",
    borderWidth: 2,
    borderColor: "rgba(53,214,195,0.2)",
  },
  avatarInitials: {
    color: colors.primary,
    fontSize: 24,
    fontWeight: "900",
    letterSpacing: 0.6,
  },
  heroCard: {
    backgroundColor: colors.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  heroTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  heroTitle: {
    color: colors.text,
    fontSize: 19,
    fontWeight: "800",
  },
  heroSubtitle: {
    color: colors.textMuted,
    marginTop: spacing.xs,
    lineHeight: 20,
  },
  heroActions: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginHorizontal: spacing.md,
    marginBottom: spacing.md,
    gap: spacing.sm,
  },
  layoutToggle: {
    minWidth: 132,
    borderRadius: 12,
    borderColor: "rgba(53,214,195,0.24)",
    borderWidth: 1,
  },
  layoutToggleLabel: {
    fontWeight: "800",
    letterSpacing: 0.4,
  },
  heroBadge: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(53,214,195,0.14)",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.3)",
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
  },
  heroBadgeText: {
    color: colors.primary,
    marginLeft: 5,
    fontWeight: "700",
    fontSize: 12,
  },
  sosButton: {
    marginHorizontal: spacing.md,
    marginBottom: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: 12,
    shadowColor: "#ef4444",
    shadowOpacity: 0.4,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 6,
  },
  sosButtonLabel: {
    fontSize: 16,
    fontWeight: "900",
    letterSpacing: 1,
  },
  list: {
    padding: spacing.md,
    paddingBottom: spacing.xl,
  },
  listSingle: {
    alignItems: "stretch",
  },
  columnWrap: {
    gap: spacing.sm,
  },
  zoneCell: {
    flex: 1,
    minWidth: 0,
  },
  zoneCellSingle: {
    width: "100%",
    maxWidth: 1140,
    minWidth: 340,
    alignSelf: "center",
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

