import React, { useEffect, useRef } from "react";
import { Animated, Image, ScrollView, StyleSheet, View } from "react-native";
import { Button, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { colors, spacing } from "../theme";

const LOGO = require("../../assets/icon.png");

export default function LandingScreen({ navigation }) {
  const insets = useSafeAreaInsets();
  const glow = useRef(new Animated.Value(0.85)).current;

  useEffect(() => {
    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(glow, { toValue: 1, duration: 1800, useNativeDriver: true }),
        Animated.timing(glow, { toValue: 0.85, duration: 1800, useNativeDriver: true }),
      ])
    );
    pulse.start();
    return () => pulse.stop();
  }, [glow]);

  return (
    <View style={[styles.container, { paddingTop: insets.top, paddingBottom: insets.bottom }]}>
      <View style={styles.bgAccentTop} />
      <View style={styles.bgAccentBottom} />
      <MaterialCommunityIcons
        name="waves"
        size={160}
        color={`${colors.primary}12`}
        style={styles.bgWave}
      />

      <ScrollView
        contentContainerStyle={styles.content}
        bounces={false}
        showsVerticalScrollIndicator={false}
      >
        <Animated.View style={[styles.logoWrap, { transform: [{ scale: glow }] }]}>
          <View style={styles.logoGlow} />
          <Image source={LOGO} style={styles.logo} resizeMode="contain" accessibilityLabel="CoastVision logo" />
        </Animated.View>

        <Text style={styles.brandTitle}>COASTVISION</Text>
        <Text style={styles.tagline}>AI Powered Beach Surveillance & Lifeguard Safety</Text>
        <Text style={styles.description}>
          Real-time monitoring, alerts, analytics and zone management.
        </Text>

        <View style={styles.featureRow}>
          <FeatureChip icon="cctv" label="Live zones" />
          <FeatureChip icon="bell-ring" label="Alerts" />
          <FeatureChip icon="chart-line" label="Analytics" />
        </View>

        <Button
          mode="contained"
          onPress={() => navigation.navigate("SignIn")}
          style={styles.cta}
          contentStyle={styles.ctaContent}
          labelStyle={styles.ctaLabel}
        >
          Get Started
        </Button>
      </ScrollView>
    </View>
  );
}

function FeatureChip({ icon, label }) {
  return (
    <View style={styles.chip}>
      <MaterialCommunityIcons name={icon} size={18} color={colors.primary} />
      <Text style={styles.chipLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  bgAccentTop: {
    position: "absolute",
    top: -80,
    right: -60,
    width: 220,
    height: 220,
    borderRadius: 110,
    backgroundColor: `${colors.primary}14`,
  },
  bgAccentBottom: {
    position: "absolute",
    bottom: -40,
    left: -80,
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: `${colors.primaryDark}10`,
  },
  bgWave: {
    position: "absolute",
    bottom: 48,
    alignSelf: "center",
    opacity: 0.35,
  },
  content: {
    flexGrow: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.xl,
  },
  logoWrap: {
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.lg,
  },
  logoGlow: {
    position: "absolute",
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: `${colors.primary}22`,
    borderWidth: 1,
    borderColor: `${colors.primary}44`,
  },
  logo: {
    width: 112,
    height: 112,
    borderRadius: 28,
  },
  brandTitle: {
    color: colors.primary,
    fontSize: 32,
    fontWeight: "800",
    letterSpacing: 3,
    textAlign: "center",
    marginBottom: spacing.sm,
  },
  tagline: {
    color: colors.text,
    fontSize: 15,
    fontWeight: "600",
    textAlign: "center",
    lineHeight: 22,
    marginBottom: spacing.md,
    paddingHorizontal: spacing.sm,
  },
  description: {
    color: colors.textMuted,
    fontSize: 14,
    textAlign: "center",
    lineHeight: 21,
    marginBottom: spacing.lg,
    paddingHorizontal: spacing.md,
  },
  featureRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: spacing.sm,
    marginBottom: spacing.xl,
  },
  chip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 20,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  chipLabel: {
    color: colors.textMuted,
    fontSize: 12,
    fontWeight: "600",
  },
  cta: {
    width: "100%",
    maxWidth: 320,
    borderRadius: 12,
    backgroundColor: colors.primary,
  },
  ctaContent: {
    height: 52,
  },
  ctaLabel: {
    color: colors.background,
    fontSize: 16,
    fontWeight: "800",
    letterSpacing: 0.5,
  },
});
