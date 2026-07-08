import React, { useEffect, useState, useRef } from "react";
import { Image, StyleSheet, View } from "react-native";
import { ActivityIndicator, Text } from "react-native-paper";
import { colors } from "../theme";

/**
 * Real-time zone streaming that starts immediately and stays live without blanking the card.
 */
export default function ZoneStream({ frameUrl, height = 200, fps = 5 }) {
  const [displayUrl, setDisplayUrl] = useState(null);
  const [failed, setFailed] = useState(false);
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const frameCountRef = useRef(0);
  const pollingIntervalRef = useRef(null);
  const urlRef = useRef(frameUrl);
  const displayUrlRef = useRef(null);
  const effectiveFps = Math.min(20, Math.max(3, Number(fps) || 10));

  const buildFrameUrl = (sourceUrl, frameIndex = 0) => {
    const separator = sourceUrl.includes("?") ? "&" : "?";
    return `${sourceUrl}${separator}_f=${frameIndex}`;
  };

  // Update URL reference
  useEffect(() => {
    urlRef.current = frameUrl;
  }, [frameUrl]);

  useEffect(() => {
    if (!frameUrl) {
      setFailed(false);
      setIsInitialLoading(false);
      setDisplayUrl(null);
      displayUrlRef.current = null;
      return undefined;
    }

    let active = true;
    setFailed(false);
    setIsInitialLoading(true);
    frameCountRef.current = 0;
    const firstFrameUrl = buildFrameUrl(frameUrl, 0);
    setDisplayUrl(firstFrameUrl);
    displayUrlRef.current = firstFrameUrl;
    setIsInitialLoading(false);

    const preloadFrame = async (candidateUrl) => {
      try {
        await Image.prefetch(candidateUrl);
        if (!active) return;
      } catch {
        if (!active) return;
        setFailed(true);
      }
    };

    preloadFrame(firstFrameUrl);

    const startPolling = setTimeout(() => {
      pollingIntervalRef.current = setInterval(() => {
        frameCountRef.current += 1;
        const newFrameUrl = buildFrameUrl(urlRef.current, frameCountRef.current);
        if (displayUrlRef.current === newFrameUrl) return;
        Image.prefetch(newFrameUrl)
          .then(() => {
            if (!active || displayUrlRef.current === newFrameUrl) return;
            setDisplayUrl(newFrameUrl);
            displayUrlRef.current = newFrameUrl;
          })
          .catch(() => {});
      }, Math.max(120, 1000 / effectiveFps));
    }, 80);

    return () => {
      active = false;
      clearTimeout(startPolling);
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [frameUrl, effectiveFps]);

  if (!frameUrl) {
    return (
      <View style={[styles.container, { height }]}>
        <Text style={styles.fallback}>No stream URL</Text>
      </View>
    );
  }

  if (failed && !displayUrl) {
    return (
      <View style={[styles.container, { height }]}>
        <Text style={styles.fallback}>Stream unavailable</Text>
      </View>
    );
  }

  return (
    <View style={[styles.container, { height }]}>
      {displayUrl && (
        <Image
          source={{ uri: displayUrl }}
          style={styles.video}
          resizeMode="cover"
          onError={() => setFailed(true)}
        />
      )}
      {!displayUrl && isInitialLoading && (
        <View style={styles.loaderOverlay}>
          <ActivityIndicator size="small" color={colors.primary} />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.surfaceAlt,
    overflow: "hidden",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    position: "relative",
  },
  video: {
    width: "100%",
    height: "100%",
  },
  loaderOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.32)",
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: "flex-end",
    alignItems: "flex-start",
    padding: 8,
    backgroundColor: "rgba(0,0,0,0.16)",
  },
  overlayText: {
    color: "#fff",
    fontSize: 11,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.8,
  },
  fallback: {
    color: colors.textMuted,
    textAlign: "center",
    marginTop: 90,
    fontWeight: "600",
  },
});

