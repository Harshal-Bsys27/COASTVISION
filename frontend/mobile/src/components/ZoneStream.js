import React, { useEffect, useState } from "react";
import { Image, StyleSheet, View } from "react-native";
import { ActivityIndicator, Text } from "react-native-paper";
import { colors } from "../theme";

/**
 * Expo Go-safe stream: polls frame.jpg from the backend.
 * Avoids expo-av / react-native-video native modules that break in Expo Go.
 */
export default function ZoneStream({ frameUrl, height = 200 }) {
  const [frameNonce, setFrameNonce] = useState(Date.now());
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!frameUrl) return undefined;
    setFailed(false);
    const timer = setInterval(() => setFrameNonce(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [frameUrl]);

  if (!frameUrl) {
    return (
      <View style={[styles.container, { height }]}>
        <Text style={styles.fallback}>No stream URL</Text>
      </View>
    );
  }

  if (failed) {
    return (
      <View style={[styles.container, { height }]}>
        <Text style={styles.fallback}>Stream unavailable</Text>
      </View>
    );
  }

  return (
    <View style={[styles.container, { height }]}>
      <Image
        source={{ uri: `${frameUrl}${frameUrl.includes("?") ? "&" : "?"}nonce=${frameNonce}` }}
        style={styles.video}
        resizeMode="cover"
        onError={() => setFailed(true)}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.surfaceAlt,
    overflow: "hidden",
    borderRadius: 8,
  },
  video: {
    width: "100%",
    height: "100%",
  },
  fallback: {
    color: colors.textMuted,
    textAlign: "center",
    marginTop: 80,
  },
});
