import React, { useEffect, useState } from "react";
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, View, useWindowDimensions } from "react-native";
import { ActivityIndicator, Button, Snackbar, Text, TextInput } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useApiContext } from "../context/ApiContext";
import { colors, spacing } from "../theme";

export default function SignInScreen() {
  const insets = useSafeAreaInsets();
  const { width } = useWindowDimensions();
  const isWide = width >= 920;
  const { baseUrl, ready, authReady, login } = useApiContext();
  const [urlInput, setUrlInput] = useState(baseUrl);
  const [phoneInput, setPhoneInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    setUrlInput(baseUrl);
  }, [baseUrl]);

  const handleSignIn = async () => {
    setLoading(true);
    setMessage("");
    try {
      await login(phoneInput, urlInput);
    } catch (err) {
      setMessage(err.message || "Sign in failed");
    } finally {
      setLoading(false);
    }
  };

  if (!ready || !authReady) {
    return (
      <View style={[styles.centered, { paddingTop: insets.top }]}>
        <ActivityIndicator color={colors.primary} />
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={[styles.container, { paddingTop: insets.top }]}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <View style={styles.bgOrbTop} />
      <View style={styles.bgOrbBottom} />
      <View style={styles.bgBeamOne} />
      <View style={styles.bgBeamTwo} />
      <ScrollView contentContainerStyle={[styles.content, isWide && styles.contentWide]} keyboardShouldPersistTaps="handled">
        <View style={styles.brandBlock}>
          <View style={styles.logoWrap}>
            <MaterialCommunityIcons name="waves" size={26} color={colors.primary} />
          </View>
          <Text variant="headlineMedium" style={styles.title}>
            CoastVision
          </Text>
          <Text style={styles.subtitle}>Lifeguard Console</Text>
          <Text style={styles.help}>
            Securely connect to your patrol feed and monitor assigned zones in real time.
          </Text>
        </View>

        <View style={[styles.formCard, isWide && styles.formCardWide]}>
          <Text style={styles.formTitle}>Sign In</Text>
          <TextInput
            label="Server URL"
            value={urlInput}
            onChangeText={setUrlInput}
            mode="outlined"
            autoCapitalize="none"
            autoCorrect={false}
            style={styles.input}
            textColor={colors.text}
            outlineColor={colors.border}
            activeOutlineColor={colors.primary}
            placeholder="http://192.168.1.4:8000"
          />

          <TextInput
            label="Phone number"
            value={phoneInput}
            onChangeText={setPhoneInput}
            mode="outlined"
            keyboardType="phone-pad"
            style={styles.input}
            textColor={colors.text}
            outlineColor={colors.border}
            activeOutlineColor={colors.primary}
            placeholder="9876543210"
          />

          <Button
            mode="contained"
            onPress={handleSignIn}
            loading={loading}
            disabled={loading}
            style={styles.button}
            contentStyle={styles.buttonContent}
            labelStyle={styles.buttonLabel}
          >
            Enter Command Center
          </Button>
        </View>
      </ScrollView>

      <Snackbar visible={Boolean(message)} onDismiss={() => setMessage("")} duration={4000}>
        {message}
      </Snackbar>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  bgOrbTop: {
    position: "absolute",
    top: -80,
    right: -60,
    width: 220,
    height: 220,
    borderRadius: 110,
    backgroundColor: "rgba(53,214,195,0.12)",
  },
  bgOrbBottom: {
    position: "absolute",
    bottom: -90,
    left: -80,
    width: 300,
    height: 300,
    borderRadius: 150,
    backgroundColor: "rgba(24,52,77,0.42)",
  },
  bgBeamOne: {
    position: "absolute",
    top: 130,
    left: -90,
    width: 300,
    height: 300,
    borderRadius: 28,
    transform: [{ rotate: "22deg" }],
    backgroundColor: "rgba(53,214,195,0.05)",
  },
  bgBeamTwo: {
    position: "absolute",
    top: 220,
    right: -120,
    width: 340,
    height: 340,
    borderRadius: 36,
    transform: [{ rotate: "-16deg" }],
    backgroundColor: "rgba(59,130,246,0.06)",
  },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: spacing.xl,
    flexGrow: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  contentWide: {
    paddingHorizontal: spacing.xl,
  },
  brandBlock: {
    alignItems: "center",
    marginBottom: spacing.lg,
    width: "100%",
    maxWidth: 760,
  },
  logoWrap: {
    width: 64,
    height: 64,
    borderRadius: 20,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(53,214,195,0.14)",
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.35)",
    marginBottom: spacing.sm,
  },
  title: {
    color: colors.primary,
    fontWeight: "800",
    marginBottom: 2,
    letterSpacing: 0.4,
  },
  subtitle: {
    color: colors.text,
    fontSize: 17,
    fontWeight: "700",
    marginBottom: spacing.xs,
  },
  help: {
    color: colors.textMuted,
    lineHeight: 20,
    textAlign: "center",
    maxWidth: 340,
  },
  formCard: {
    width: "100%",
    maxWidth: 430,
    backgroundColor: colors.surface,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    shadowColor: "#000",
    shadowOpacity: 0.22,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    elevation: 4,
  },
  formCardWide: {
    maxWidth: 560,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.lg,
  },
  formTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: "800",
    marginBottom: spacing.sm,
  },
  input: {
    backgroundColor: "rgba(6,19,31,0.52)",
    marginBottom: spacing.md,
  },
  button: {
    marginTop: spacing.xs,
    borderRadius: 12,
    backgroundColor: colors.primary,
  },
  buttonContent: {
    height: 46,
  },
  buttonLabel: {
    color: colors.background,
    fontWeight: "800",
    fontSize: 14,
  },
});
