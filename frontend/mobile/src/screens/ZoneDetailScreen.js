import React, { useEffect, useState } from "react";
import { ScrollView, StyleSheet, View } from "react-native";
import { ActivityIndicator, Button, Text } from "react-native-paper";
import { useNavigation } from "@react-navigation/native";
import ZoneStream from "../components/ZoneStream";
import { useApiContext } from "../context/ApiContext";
import { usePollApi } from "../hooks/usePollApi";
import { POLL_DETECTIONS_MS } from "../shared/constants";
import { formatConfidence } from "../utils/format";
import { isZoneAllowed } from "../utils/zoneFilter";
import { colors, layout, spacing } from "../theme";

export default function ZoneDetailScreen({ route }) {
  const navigation = useNavigation();
  const { zone } = route.params;
  const { api, assignedZones } = useApiContext();
  const [detections, setDetections] = useState({ count: 0, items: [] });
  const allowed = isZoneAllowed(zone.id, assignedZones);

  const detectionsPoll = usePollApi(
    () => api.detections(zone.id),
    POLL_DETECTIONS_MS,
    allowed,
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
      <Text variant="headlineSmall" style={styles.title}>
        {zone.name || `Zone ${zone.id}`}
      </Text>

      <ZoneStream
        frameUrl={api.frameUrl(zone.id)}
        height={layout.zoneCardHeight + 40}
      />

      <View style={styles.statsCard}>
        <Text style={styles.statsTitle}>Live Detections</Text>
        <Text style={styles.statsValue}>{detections.count ?? 0} people</Text>
      </View>

      {detectionsPoll.loading && detections.items.length === 0 ? (
        <ActivityIndicator color={colors.primary} style={{ marginTop: spacing.md }} />
      ) : (
        detections.items.map((item, index) => (
          <View key={`${item.label}-${index}`} style={styles.detectionRow}>
            <Text style={styles.detectionLabel}>{item.label || "object"}</Text>
            <Text style={styles.detectionMeta}>{formatConfidence(item.conf)}</Text>
          </View>
        ))
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
  title: {
    color: colors.text,
    fontWeight: "800",
    marginBottom: spacing.md,
  },
  statsCard: {
    marginTop: spacing.md,
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: spacing.md,
  },
  statsTitle: {
    color: colors.textMuted,
    marginBottom: spacing.xs,
  },
  statsValue: {
    color: colors.primary,
    fontSize: 24,
    fontWeight: "800",
  },
  detectionRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    backgroundColor: colors.surface,
    borderRadius: 8,
    padding: spacing.md,
    marginTop: spacing.sm,
  },
  detectionLabel: {
    color: colors.text,
    fontWeight: "600",
  },
  detectionMeta: {
    color: colors.textMuted,
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
