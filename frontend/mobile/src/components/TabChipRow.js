import React from "react";
import { ScrollView, StyleSheet, View } from "react-native";
import { Chip } from "react-native-paper";
import { colors, layout, spacing } from "../theme";

/**
 * Fixed-height horizontal tab chip row.
 * Prevents chips from stretching when sibling content uses flex layouts.
 */
export default function TabChipRow({ tabs, activeId, onSelect }) {
  return (
    <View style={styles.bar}>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        bounces={false}
        style={styles.scroll}
        contentContainerStyle={styles.content}
      >
        {tabs.map((tab) => (
          <Chip
            key={tab.id}
            compact
            selected={activeId === tab.id}
            onPress={() => onSelect(tab.id)}
            style={[styles.chip, activeId === tab.id && styles.chipSelected]}
            textStyle={styles.chipText}
          >
            {tab.label}
          </Chip>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  bar: {
    height: layout.tabChipBarHeight,
    flexGrow: 0,
    flexShrink: 0,
    backgroundColor: colors.background,
  },
  scroll: {
    flexGrow: 0,
  },
  content: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: spacing.md,
    minHeight: layout.tabChipBarHeight,
    gap: spacing.sm,
  },
  chip: {
    height: layout.tabChipHeight,
    alignSelf: "center",
    justifyContent: "center",
    backgroundColor: colors.surface,
    marginRight: spacing.sm,
  },
  chipSelected: {
    backgroundColor: colors.surfaceAlt,
  },
  chipText: {
    color: colors.text,
    fontSize: 13,
    lineHeight: 16,
    marginVertical: 0,
  },
});
