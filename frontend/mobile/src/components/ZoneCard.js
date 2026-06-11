import React, { memo } from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { Card, Chip, Text } from "react-native-paper";
import ZoneStream from "./ZoneStream";
import { ZONE_COLORS } from "../shared/constants";
import { colors, layout, spacing } from "../theme";

function ZoneCard({ zone, personCount, frameUrl, onPress, index = 0 }) {
  const accent = ZONE_COLORS[index % ZONE_COLORS.length];

  return (
    <Pressable onPress={onPress}>
      <Card style={styles.card}>
        <View style={[styles.header, { borderLeftColor: accent }]}>
          <Text variant="titleMedium" style={styles.title}>
            {zone.name || `Zone ${zone.id}`}
          </Text>
          <Chip compact textStyle={styles.chipText} style={styles.chip}>
            {personCount ?? 0} people
          </Chip>
        </View>
        <ZoneStream frameUrl={frameUrl} height={layout.zoneCardHeight - 56} />
      </Card>
    </Pressable>
  );
}

export default memo(ZoneCard);

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    marginBottom: spacing.md,
    borderRadius: 12,
    overflow: "hidden",
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderLeftWidth: 4,
  },
  title: {
    color: colors.text,
    fontWeight: "700",
  },
  chip: {
    backgroundColor: colors.surfaceAlt,
  },
  chipText: {
    color: colors.primary,
    fontWeight: "700",
  },
});
