import React, { useEffect, useState, useRef } from "react";
import { Image, StyleSheet, View, useWindowDimensions } from "react-native";
import { Text, ActivityIndicator } from "react-native-paper";
import { colors } from "../theme";

/**
 * Flicker-free zone video player for detail view.
 * Never blanks screen - keeps showing previous frame while loading new one.
 * Ultra-smooth streaming experience.
 */
export default function ZoneVideoPlayer({ 
  frameUrl, 
  streamUrl,
  height,
  aspectRatio = 16 / 9,
  fps = 5,
}) {
  const { width } = useWindowDimensions();
  const [displayUrl, setDisplayUrl] = useState(null);
  const [failed, setFailed] = useState(false);
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const pollingIntervalRef = useRef(null);
  const frameCountRef = useRef(0);
  const requestIdRef = useRef(0);
  const urlRef = useRef(frameUrl || streamUrl);
  const displayUrlRef = useRef(null);
  const effectiveFps = Math.min(20, Math.max(3, Number(fps) || 10));

  // Calculate responsive dimensions
  const containerWidth = width - 32; // padding
  let calculatedHeight = height;
  if (!calculatedHeight) {
    calculatedHeight = Math.round(containerWidth / aspectRatio);
  }

  // Update URL reference
  useEffect(() => {
    urlRef.current = frameUrl || streamUrl;
  }, [frameUrl, streamUrl]);

  useEffect(() => {
    const activeUrl = frameUrl || streamUrl;
    if (!activeUrl) {
      setFailed(false);
      setIsInitialLoading(false);
      return undefined;
    }

    let active = true;
    setFailed(false);
    setIsInitialLoading(true);
    frameCountRef.current = 0;
    requestIdRef.current += 1;
    const firstFrameUrl = `${activeUrl}${activeUrl.includes("?") ? "&" : "?"}_f=0`;
    setDisplayUrl(firstFrameUrl);
    displayUrlRef.current = firstFrameUrl;

    Image.prefetch(firstFrameUrl)
      .then(() => {
        if (!active) return;
        setIsInitialLoading(false);
        setFailed(false);
      })
      .catch(() => {
        if (!active) return;
        setIsInitialLoading(false);
        setFailed(true);
      });

    const startPolling = setTimeout(() => {
      pollingIntervalRef.current = setInterval(() => {
        frameCountRef.current += 1;
        const newFrameUrl = `${urlRef.current}${urlRef.current.includes("?") ? "&" : "?"}_f=${frameCountRef.current}`;
        const requestId = ++requestIdRef.current;

        Image.prefetch(newFrameUrl)
          .then(() => {
            if (!active || requestId !== requestIdRef.current) return;
            setDisplayUrl(newFrameUrl);
            displayUrlRef.current = newFrameUrl;
          })
          .catch(() => {});
      }, Math.max(80, 1000 / effectiveFps));
    }, 120);

    return () => {
      active = false;
      clearTimeout(startPolling);
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [frameUrl, streamUrl, effectiveFps]);

  if (!frameUrl && !streamUrl) {
    return (
      <View style={[styles.container, { height: calculatedHeight }]}>
        <Text style={styles.fallback}>No stream URL configured</Text>
      </View>
    );
  }

  // Show error only if initial frame failed to load
  if (failed && !displayUrl) {
    return (
      <View style={[styles.container, { height: calculatedHeight }]}>
        <Text style={styles.fallback}>Video stream unavailable</Text>
      </View>
    );
  }

  return (
    <View style={[styles.container, { height: calculatedHeight }]}>
      {displayUrl && (
        <Image
          source={{ uri: displayUrl }}
          style={styles.video}
          resizeMode="cover"
        />
      )}
      {isInitialLoading && (
        <View style={styles.loader}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={styles.loaderText}>Loading stream...</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: "100%",
    backgroundColor: colors.surfaceAlt,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: "hidden",
    position: "relative",
    justifyContent: "center",
    alignItems: "center",
  },
  video: {
    width: "100%",
    height: "100%",
  },
  loader: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.4)",
    zIndex: 1,
  },
  loaderText: {
    color: colors.primary,
    marginTop: 12,
    fontSize: 12,
    fontWeight: "600",
  },
  fallback: {
    color: colors.textMuted,
    textAlign: "center",
    fontWeight: "600",
    fontSize: 14,
  },
});

