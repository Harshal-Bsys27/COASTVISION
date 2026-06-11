import React from "react";
import { StyleSheet } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { colors } from "../theme";

/**
 * Top/side safe area for screens without a native stack header.
 * Bottom inset is handled by the tab bar in RootNavigator.
 */
export default function ScreenContainer({ children, style }) {
  return (
    <SafeAreaView style={[styles.container, style]} edges={["top", "left", "right"]}>
      {children}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
});
