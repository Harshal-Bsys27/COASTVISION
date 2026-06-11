import React, { useEffect, useState } from "react";
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, View } from "react-native";
import { ActivityIndicator, Button, Snackbar, Text, TextInput } from "react-native-paper";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useApiContext } from "../context/ApiContext";
import { colors, spacing } from "../theme";

export default function SignInScreen() {
  const insets = useSafeAreaInsets();
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
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <Text variant="headlineMedium" style={styles.title}>
          CoastVision
        </Text>
        <Text style={styles.subtitle}>Lifeguard sign in</Text>
        <Text style={styles.help}>
          Your admin must create your account on the web dashboard first. Sign in with your phone number.
        </Text>

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
        >
          Sign In
        </Button>
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
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: spacing.xl,
  },
  title: {
    color: colors.primary,
    fontWeight: "800",
    marginBottom: spacing.xs,
  },
  subtitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: "700",
    marginBottom: spacing.sm,
  },
  help: {
    color: colors.textMuted,
    lineHeight: 20,
    marginBottom: spacing.lg,
  },
  input: {
    backgroundColor: colors.surface,
    marginBottom: spacing.md,
  },
  button: {
    marginTop: spacing.sm,
    borderRadius: 10,
  },
});
