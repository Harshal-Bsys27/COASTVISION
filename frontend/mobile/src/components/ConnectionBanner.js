import React from "react";
import { StyleSheet, View } from "react-native";
import { Text } from "react-native-paper";
import { colors, spacing } from "../theme";

export default function ConnectionBanner({ visible, message }) {
  if (!visible) return null;

  return (
    <View style={styles.banner}>
      <Text style={styles.text}>
        {message || "Cannot reach server — check Wi-Fi and URL in Settings"}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    backgroundColor: "rgba(239,68,68,0.18)",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(239,68,68,0.32)",
  },
  text: {
    color: colors.text,
    fontSize: 13,
    fontWeight: "600",
  },
});
