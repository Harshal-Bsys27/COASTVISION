import React, { memo } from "react";
import { Image, StyleSheet, View } from "react-native";
import { Card, Text } from "react-native-paper";
import { formatConfidence, formatTimestamp, getSeverityColor } from "../utils/format";
import { colors, spacing } from "../theme";

function AlertCard({ alert, imageUrl }) {
  const label = alert?.label || alert?.class || "Alert";
  const severityColor = getSeverityColor(label, colors);

  return (
    <Card style={[styles.card, { borderLeftColor: severityColor }]}>
      <View style={styles.row}>
        <View style={styles.content}>
          <Text variant="titleSmall" style={[styles.label, { color: severityColor }]}>
            {label}
          </Text>
          <Text style={styles.meta}>Zone {alert?.zone ?? "—"}</Text>
          <Text style={styles.meta}>Confidence: {formatConfidence(alert?.conf)}</Text>
          <Text style={styles.meta}>{formatTimestamp(alert?.ts_utc || alert?.timestamp)}</Text>
        </View>
        {imageUrl ? (
          <Image source={{ uri: imageUrl }} style={styles.thumbnail} resizeMode="cover" />
        ) : null}
      </View>
    </Card>
  );
}

export default memo(AlertCard);

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    marginBottom: spacing.sm,
    borderLeftWidth: 4,
    borderRadius: 10,
  },
  row: {
    flexDirection: "row",
    padding: spacing.md,
    gap: spacing.md,
  },
  content: {
    flex: 1,
  },
  label: {
    fontWeight: "800",
    marginBottom: spacing.xs,
  },
  meta: {
    color: colors.textMuted,
    fontSize: 13,
    marginBottom: 2,
  },
  thumbnail: {
    width: 72,
    height: 72,
    borderRadius: 8,
    backgroundColor: colors.surfaceAlt,
  },
});
