import React, { useEffect, useState } from "react";
import { ScrollView, StyleSheet, View, useWindowDimensions } from "react-native";
import { ActivityIndicator, Button, Text } from "react-native-paper";
import { useNavigation } from "@react-navigation/native";
import ZoneVideoPlayer from "../components/ZoneVideoPlayer";
import { useApiContext } from "../context/ApiContext";
import { usePollApi } from "../hooks/usePollApi";
import { POLL_DETECTIONS_MS } from "../shared/constants";
import { formatConfidence } from "../utils/format";
import { isZoneAllowed } from "../utils/zoneFilter";
import { colors, layout, spacing } from "../theme";

export default function ZoneDetailScreen({ route }) {
  const navigation = useNavigation();
  const { width } = useWindowDimensions();
  const zoneParam = route?.params?.zone ?? route?.params ?? {};
  const zoneId = zoneParam?.id ?? zoneParam?.zone ?? route?.params?.zoneId ?? null;
  const zoneName = zoneParam?.name ?? zoneParam?.zoneName ?? (zoneId ? `Zone ${zoneId}` : "Zone");
  const zone = zoneParam && typeof zoneParam === "object" ? { ...zoneParam, id: zoneId, name: zoneName } : { id: zoneId, name: zoneName };
  const { api, assignedZones, streamQuality } = useApiContext();
  const [detections, setDetections] = useState({ count: 0, items: [] });
  const [detailReady, setDetailReady] = useState(Boolean(zoneId));
  const allowed = isZoneAllowed(zoneId, assignedZones);

  // Calculate responsive video height based on screen width
  // Use 16:9 aspect ratio for most video content
  const videoContainerWidth = Math.max(320, width - spacing.md * 2);
  const videoHeight = Math.round(videoContainerWidth / (16 / 9));

  const detectionsPoll = usePollApi(
    () => (zoneId ? api.detections(zoneId) : Promise.resolve({ count: 0, items: [] })),
    POLL_DETECTIONS_MS,
    allowed && Boolean(zoneId),
    { count: 0, items: [] }
  );

  useEffect(() => {
    if (!allowed) {
      navigation.goBack();
    }
  }, [allowed, navigation]);

  useEffect(() => {
    if (detectionsPoll.data) {
      setDetections(detectionsPoll.data);
    }
  }, [detectionsPoll.data]);

  useEffect(() => {
    setDetailReady(Boolean(zoneId));
  }, [zoneId]);

  useEffect(() => {
    if (allowed && zoneId) {
      detectionsPoll.refresh?.();
    }
  }, [allowed, zoneId, detectionsPoll.refresh]);

  if (!allowed) {
    return (
      <View style={styles.blocked}>
        <Text style={styles.blockedTitle}>Zone not assigned</Text>
        <Text style={styles.blockedText}>You do not have access to this zone.</Text>
        <Button mode="contained" onPress={() => navigation.goBack()} style={{ marginTop: spacing.md }}>
          Go Back
        </Button>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.pageHeader}>
        <Text style={styles.pageTitle}>Zone Command Suite</Text>
        <Text style={styles.pageSubtitle}>Dedicated control for this lifeguard zone, live video, and detection analytics.</Text>
      </View>
      <View style={styles.heroCard}>
        <Text style={styles.heroEyebrow}>Zone Monitor</Text>
        <Text variant="headlineSmall" style={styles.title}>
          {zone.name || zoneName || `Zone ${zoneId}`}
        </Text>
        {zoneId ? (
          <Text style={styles.subtitle}>Zone ID {zoneId}</Text>
        ) : (
          <Text style={styles.subtitle}>The selected zone is not available yet.</Text>
        )}
      </View>

      <View style={styles.detailsCard}>
        <View style={styles.detailsRow}>
          <Text style={styles.detailsTitle}>Live zone intelligence</Text>
          <Text style={styles.detailsBadge}>Premium</Text>
        </View>
        <Text style={styles.detailsText}>
          Real-time CCTV feed, people counts, and detections to support faster lifeguard decisions.
        </Text>
      </View>

      <ZoneVideoPlayer
        frameUrl={zoneId ? api.frameUrl(zoneId, 640) : null}
        streamUrl={zoneId ? api.mjpegUrl(zoneId) : null}
        height={videoHeight}
        aspectRatio={16 / 9}
        fps={Math.max(4, Number(streamQuality) || 6)}
      />

      <View style={styles.statsCard}>
        <View style={styles.statValueWrap}>
          <Text style={styles.statsValue}>{detections.count ?? 0}</Text>
          <Text style={styles.statsTitle}>People Detected</Text>
        </View>
        <Text style={styles.statsDescription}>Real-time crowd monitoring for this zone</Text>
      </View>

      {detectionsPoll.loading && detections.items.length === 0 ? (
        <ActivityIndicator color={colors.primary} style={{ marginTop: spacing.md }} />
      ) : detections.items.length > 0 ? (
        <View>
          <Text style={[styles.title, { marginTop: spacing.md, marginBottom: spacing.md }]}>
            Live Objects
          </Text>
          {detections.items.map((item, index) => (
            <View key={`${item.label}-${index}`} style={styles.detectionRow}>
              <View style={styles.detectionInfo}>
                <Text style={styles.detectionLabel}>{item.label || "object"}</Text>
                <Text style={styles.detectionMeta}>Confidence</Text>
              </View>
              <Text style={styles.detectionConfidence}>{formatConfidence(item.conf)}</Text>
            </View>
          ))}
        </View>
      ) : (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateText}>No detections at this moment</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.md,
    paddingBottom: spacing.xl,
  },
  heroCard: {
    backgroundColor: colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  pageHeader: {
    backgroundColor: colors.surface,
    padding: spacing.md,
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
  heroEyebrow: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 2,
  },
  title: {
    color: colors.text,
    fontWeight: "900",
    fontSize: 20,
    marginBottom: spacing.xs,
    letterSpacing: 0.2,
  },
  subtitle: {
    color: colors.textMuted,
    fontSize: 13,
    fontWeight: "500",
    marginBottom: spacing.md,
  },
  detailsCard: {
    backgroundColor: "rgba(53,214,195,0.08)",
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.24)",
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  detailsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: spacing.xs,
  },
  detailsTitle: {
    color: colors.text,
    fontWeight: "900",
    fontSize: 14,
  },
  detailsBadge: {
    color: colors.primary,
    fontWeight: "900",
    fontSize: 12,
    letterSpacing: 0.8,
    textTransform: "uppercase",
  },
  detailsText: {
    color: colors.textMuted,
    lineHeight: 20,
    fontSize: 13,
  },
  statsCard: {
    marginTop: spacing.lg,
    backgroundColor: "rgba(16,34,53,0.68)",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.24)",
    padding: spacing.lg,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
    elevation: 3,
  },
  statValueWrap: {
    marginBottom: spacing.md,
  },
  statsTitle: {
    color: colors.textMuted,
    fontSize: 12,
    fontWeight: "600",
    marginTop: 4,
    letterSpacing: 0.5,
  },
  statsValue: {
    color: colors.primary,
    fontSize: 36,
    fontWeight: "900",
    letterSpacing: -0.5,
  },
  statsDescription: {
    color: "rgba(226,232,240,0.6)",
    fontSize: 13,
    fontWeight: "500",
  },
  detectionRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: "rgba(16,34,53,0.68)",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.22)",
    borderLeftWidth: 4,
    borderLeftColor: colors.primary,
    padding: spacing.md,
    marginTop: spacing.sm,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
  detectionInfo: {
    flex: 1,
  },
  detectionLabel: {
    color: colors.text,
    fontWeight: "900",
    fontSize: 15.5,
    letterSpacing: 0.15,
  },
  detectionMeta: {
    color: colors.textMuted,
    fontSize: 12,
    fontWeight: "600",
    marginTop: 4,
  },
  detectionConfidence: {
    color: colors.primary,
    fontWeight: "800",
    fontSize: 14,
    marginLeft: spacing.md,
  },
  emptyState: {
    marginTop: spacing.xl,
    paddingVertical: spacing.xl,
    alignItems: "center",
  },
  emptyStateText: {
    color: colors.textMuted,
    fontSize: 14,
    fontWeight: "500",
  },
  blocked: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.lg,
    backgroundColor: colors.background,
  },
  blockedTitle: {
    color: colors.text,
    fontWeight: "700",
    fontSize: 18,
    marginBottom: spacing.sm,
  },
  blockedText: {
    color: colors.textMuted,
    textAlign: "center",
  },
});

