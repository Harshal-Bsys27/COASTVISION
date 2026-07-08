import React, { useEffect, useRef } from "react";
import { Animated, ScrollView, StyleSheet, View, useWindowDimensions } from "react-native";
import { Button, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { colors, spacing } from "../theme";

export default function LandingScreen({ navigation }) {
  const insets = useSafeAreaInsets();
  const { width } = useWindowDimensions();
  const isWide = width >= 920;
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
      <View style={styles.bgBeamOne} />
      <View style={styles.bgBeamTwo} />
      <MaterialCommunityIcons
        name="waves"
        size={220}
        color={`${colors.primary}12`}
        style={styles.bgWave}
      />

      <ScrollView
        contentContainerStyle={[styles.content, isWide && styles.contentWide]}
        bounces={false}
        showsVerticalScrollIndicator={false}
      >
        <View style={[styles.heroCard, isWide && styles.heroCardWide]}>
          <Animated.View style={[styles.logoWrap, { transform: [{ scale: glow }] }]}>
            <View style={styles.logoGlow} />
            <View style={styles.logo} accessibilityLabel="CoastVision logo placeholder">
              <Text style={styles.logoText}>CV</Text>
            </View>
          </Animated.View>

          <Text style={styles.brandTitle}>COASTVISION</Text>
          <Text style={styles.tagline}>AI Coastal Safety for Lifeguards</Text>
          <Text style={styles.description}>
            Monitor critical zones, detect incidents early, and respond faster with real-time intelligence.
          </Text>

          <View style={styles.featureRow}>
            <FeatureChip icon="video-wireless" label="Live Zones" />
            <FeatureChip icon="chart-line" label="Smart Analytics" />
            <FeatureChip icon="shield-alert" label="Rapid Alerts" />
          </View>

          <View style={styles.metricsRow}>
            <MetricTile icon="wave" value="24x7" label="Monitoring" />
            <MetricTile icon="radar" value="AI" label="Detection" />
            <MetricTile icon="account-group" value="Team" label="Response" />
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
        </View>
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

function MetricTile({ icon, value, label }) {
  return (
    <View style={styles.metricTile}>
      <MaterialCommunityIcons name={icon} size={18} color={colors.primary} />
      <Text style={styles.metricValue}>{value}</Text>
      <Text style={styles.metricLabel}>{label}</Text>
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
    bottom: -90,
    left: -90,
    width: 280,
    height: 280,
    borderRadius: 140,
    backgroundColor: "rgba(26,168,151,0.14)",
  },
  bgBeamOne: {
    position: "absolute",
    top: 160,
    left: -80,
    width: 280,
    height: 280,
    borderRadius: 30,
    transform: [{ rotate: "24deg" }],
    backgroundColor: "rgba(53,214,195,0.05)",
  },
  bgBeamTwo: {
    position: "absolute",
    top: 220,
    right: -120,
    width: 340,
    height: 340,
    borderRadius: 36,
    transform: [{ rotate: "-18deg" }],
    backgroundColor: "rgba(59,130,246,0.07)",
  },
  bgWave: {
    position: "absolute",
    bottom: 28,
    alignSelf: "center",
    opacity: 0.45,
  },
  content: {
    flexGrow: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.xl,
  },
  contentWide: {
    paddingHorizontal: spacing.xl,
  },
  heroCard: {
    width: "100%",
    maxWidth: 460,
    borderRadius: 26,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.xl,
    backgroundColor: "rgba(16,34,53,0.82)",
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.24)",
    shadowColor: "#000",
    shadowOpacity: 0.3,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 10 },
    elevation: 6,
    alignItems: "center",
  },
  heroCardWide: {
    maxWidth: 760,
    paddingHorizontal: spacing.xl,
  },
  logoWrap: {
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.md,
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
    backgroundColor: colors.surface,
    alignItems: "center",
    justifyContent: "center",
  },
  logoText: {
    color: colors.primary,
    fontSize: 36,
    fontWeight: "900",
  },
  brandTitle: {
    color: colors.primary,
    fontSize: 34,
    fontWeight: "900",
    letterSpacing: 3.4,
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
    fontSize: 14.5,
    textAlign: "center",
    lineHeight: 22,
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
  metricsRow: {
    width: "100%",
    flexDirection: "row",
    justifyContent: "space-between",
    gap: spacing.sm,
    marginBottom: spacing.lg,
  },
  metricTile: {
    flex: 1,
    minHeight: 86,
    borderRadius: 14,
    backgroundColor: "rgba(6,19,31,0.56)",
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.22)",
    alignItems: "center",
    justifyContent: "center",
    gap: 3,
  },
  metricValue: {
    color: colors.text,
    fontSize: 16,
    fontWeight: "800",
  },
  metricLabel: {
    color: colors.textMuted,
    fontSize: 11,
    fontWeight: "700",
    letterSpacing: 0.35,
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
    maxWidth: 360,
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

