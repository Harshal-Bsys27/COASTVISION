import React, { useEffect, useState, useRef } from "react";
import { Image, StyleSheet, View } from "react-native";
import { ActivityIndicator, Text } from "react-native-paper";
import { colors } from "../theme";

/**
 * Flicker-free frame streaming for zone cards.
 * Smart strategy: Keep showing previous frame while new one loads.
 * Never blank the screen, no flickering.
 */
export default function ZoneStream({ frameUrl, height = 200, fps = 5 }) {
  const [displayUrl, setDisplayUrl] = useState(null);
  const [failed, setFailed] = useState(false);
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const frameCountRef = useRef(0);
  const pollingIntervalRef = useRef(null);
  const urlRef = useRef(frameUrl);
  const displayUrlRef = useRef(null);
  const effectiveFps = Math.min(15, Math.max(1, Number(fps) || 5));

  // Update URL reference
  useEffect(() => {
    urlRef.current = frameUrl;
  }, [frameUrl]);

  useEffect(() => {
    if (!frameUrl) {
      setFailed(false);
      setIsInitialLoading(false);
      return undefined;
    }

    setFailed(false);
    frameCountRef.current = 0;
    setDisplayUrl(null);
    setIsInitialLoading(true);

    // Load first frame
    const firstFrameUrl = `${frameUrl}${frameUrl.includes("?") ? "&" : "?"}_f=0`;
    Image.prefetch(firstFrameUrl)
      .then(() => {
        setDisplayUrl(firstFrameUrl);
        displayUrlRef.current = firstFrameUrl;
        setIsInitialLoading(false);
      })
      .catch(() => {
        setIsInitialLoading(false);
        setFailed(true);
      });

    // Start polling after 1 second
    const startPolling = setTimeout(() => {
      pollingIntervalRef.current = setInterval(() => {
        frameCountRef.current += 1;
        const newFrameUrl = `${urlRef.current}${urlRef.current.includes("?") ? "&" : "?"}_f=${frameCountRef.current}`;
        
        // Prefetch next frame but don't block on it
        Image.prefetch(newFrameUrl).then(() => {
          setDisplayUrl(newFrameUrl);
          displayUrlRef.current = newFrameUrl;
        }).catch(() => {});
      }, 1000 / effectiveFps); // configurable FPS polling
    }, 1000);

    return () => {
      clearTimeout(startPolling);
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [frameUrl]);

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
        />
      )}
      {isInitialLoading && (
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
    backgroundColor: "rgba(0,0,0,0.3)",
  },
  fallback: {
    color: colors.textMuted,
    textAlign: "center",
    marginTop: 90,
    fontWeight: "600",
  },
});
