import React, { memo } from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { Card, Chip, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import ZoneStream from "./ZoneStream";
import { ZONE_COLORS } from "../shared/constants";
import { colors, layout, spacing } from "../theme";

function ZoneCard({ zone, personCount, frameUrl, onPress, index = 0, style, streamHeight, fps }) {
  const accent = ZONE_COLORS[index % ZONE_COLORS.length];

  return (
    <Pressable onPress={onPress} style={[styles.pressable, style]}>
      <Card style={[styles.card, { borderColor: `${accent}55` }]}>
        <View style={[styles.header, { borderLeftColor: accent }]}>
          <View style={styles.titleWrap}>
            <Text variant="titleMedium" style={styles.title}>
              {zone.name || `Zone ${zone.id}`}
            </Text>
            <Text style={styles.subtitle}>Live coastal feed</Text>
          </View>
          <View style={styles.badges}>
            <Chip compact textStyle={styles.chipText} style={[styles.chip, { backgroundColor: `${accent}25` }]}>
              {personCount ?? 0} people
            </Chip>
          </View>
        </View>
        <ZoneStream frameUrl={frameUrl} height={streamHeight || layout.zoneCardHeight} fps={fps} />
        <View style={styles.footer}>
          <Text style={styles.footerText}>Tap for zone details</Text>
          <MaterialCommunityIcons name="chevron-right" size={18} color={colors.textMuted} />
        </View>
      </Card>
    </Pressable>
  );
}

export default memo(ZoneCard);

const styles = StyleSheet.create({
  pressable: {
    width: "100%",
  },
  card: {
    backgroundColor: colors.surface,
    marginBottom: spacing.md,
    borderRadius: 16,
    overflow: "hidden",
    borderWidth: 1,
    shadowColor: "#000",
    shadowOpacity: 0.2,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    elevation: 4,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm + 2,
    borderLeftWidth: 4,
    backgroundColor: "rgba(255,255,255,0.02)",
  },
  titleWrap: {
    flex: 1,
    marginRight: spacing.sm,
  },
  title: {
    color: colors.text,
    fontWeight: "800",
    fontSize: 17,
  },
  subtitle: {
    color: colors.textMuted,
    fontSize: 12,
    marginTop: 2,
  },
  badges: {
    flexDirection: "row",
    alignItems: "center",
  },
  chip: {
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  chipText: {
    color: colors.text,
    fontWeight: "700",
    fontSize: 12,
  },
  footer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm + 2,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    backgroundColor: "rgba(255,255,255,0.02)",
  },
  footerText: {
    color: colors.textMuted,
    fontSize: 12,
    fontWeight: "600",
  },
});
