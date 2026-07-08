import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { createApi, isValidApiUrl, normalizeBaseUrl } from "../shared/api";
import {
  STORAGE_API_URL_KEY,
  STORAGE_LIFEGUARD_KEY,
  STORAGE_SESSION_TOKEN_KEY,
  STORAGE_STREAM_QUALITY_KEY,
} from "../shared/constants";
import { logInfo } from "../utils/logger";

const ApiContext = createContext(null);

export function ApiProvider({ children }) {
  const [baseUrl, setBaseUrl] = useState("");
  const [ready, setReady] = useState(false);
  const [authReady, setAuthReady] = useState(false);
  const [connected, setConnected] = useState(false);
  const [health, setHealth] = useState(null);
  const [sessionToken, setSessionToken] = useState("");
  const [lifeguard, setLifeguard] = useState(null);
  const [streamQuality, setStreamQuality] = useState("10");
  const sessionTokenRef = useRef("");

  useEffect(() => {
    sessionTokenRef.current = sessionToken;
  }, [sessionToken]);

  const api = useMemo(
    () => createApi(baseUrl, () => sessionTokenRef.current),
    [baseUrl]
  );

  const assignedZones = useMemo(() => {
    const zones = lifeguard?.zones;
    return Array.isArray(zones) ? zones : [];
  }, [lifeguard]);

  const persistSession = useCallback(async (token, profile) => {
    await AsyncStorage.multiSet([
      [STORAGE_SESSION_TOKEN_KEY, token || ""],
      [STORAGE_LIFEGUARD_KEY, JSON.stringify(profile || {})],
    ]);
  }, []);

  const clearSessionStorage = useCallback(async () => {
    await AsyncStorage.multiRemove([STORAGE_SESSION_TOKEN_KEY, STORAGE_LIFEGUARD_KEY]);
  }, []);

  const saveStreamQuality = useCallback(async (value) => {
    setStreamQuality(String(value));
    try {
      await AsyncStorage.setItem(STORAGE_STREAM_QUALITY_KEY, String(value));
    } catch {
      // Ignore preference save failures, still keep UI responsive.
    }
  }, []);

  useEffect(() => {
    let mounted = true;

    async function bootstrap() {
      try {
        const [[, storedUrl], [, storedToken], [, storedLifeguard], [, storedStreamQuality]] = await AsyncStorage.multiGet([
          STORAGE_API_URL_KEY,
          STORAGE_SESSION_TOKEN_KEY,
          STORAGE_LIFEGUARD_KEY,
          STORAGE_STREAM_QUALITY_KEY,
        ]);

        if (!mounted) return;

        if (storedUrl) {
          const url = normalizeBaseUrl(storedUrl);
          setBaseUrl(url);
          logInfo("Loaded saved server URL", url);
        }

        if (storedToken) {
          setSessionToken(storedToken);
          sessionTokenRef.current = storedToken;
        }

        if (storedLifeguard) {
          try {
            setLifeguard(JSON.parse(storedLifeguard));
          } catch {
            setLifeguard(null);
          }
        }

        if (storedStreamQuality) {
          setStreamQuality(storedStreamQuality);
        }

        if (storedToken && storedUrl) {
          const url = normalizeBaseUrl(storedUrl);
          const client = createApi(url, () => storedToken);
          try {
            const me = await client.lifeguardMe();
            if (mounted) {
              setLifeguard(me);
              setSessionToken(storedToken);
              logInfo("Restored lifeguard session", { name: me?.name, zones: me?.zones });
            }
          } catch {
            if (mounted) {
              setSessionToken("");
              setLifeguard(null);
              sessionTokenRef.current = "";
              await clearSessionStorage();
              logInfo("Stored session expired — sign in again");
            }
          }
        }
      } finally {
        if (mounted) {
          setReady(true);
          setAuthReady(true);
        }
      }
    }

    bootstrap();
    return () => {
      mounted = false;
    };
  }, [clearSessionStorage]);

  const saveBaseUrl = useCallback(async (url) => {
    const normalized = normalizeBaseUrl(url);
    if (!isValidApiUrl(normalized)) {
      throw new Error("Enter a valid URL starting with http:// or https://");
    }
    await AsyncStorage.setItem(STORAGE_API_URL_KEY, normalized);
    setBaseUrl(normalized);
    logInfo("Saved server URL", normalized);
    return normalized;
  }, []);

  const testConnection = useCallback(async () => {
    if (!baseUrl) {
      throw new Error("Set a server URL first");
    }
    logInfo("Testing connection", baseUrl);
    const result = await api.health();
    setConnected(true);
    setHealth(result);
    logInfo("Connection OK", { zones: result?.zones, device: result?.device });
    return result;
  }, [api, baseUrl]);

  const login = useCallback(
    async (phone, urlOverride) => {
      const phoneValue = String(phone || "").trim();
      if (!phoneValue) {
        throw new Error("Enter your phone number");
      }

      let activeUrl = baseUrl;
      if (urlOverride) {
        activeUrl = await saveBaseUrl(urlOverride);
      }
      if (!activeUrl) {
        throw new Error("Set the server URL first");
      }

      const client = createApi(activeUrl);
      const result = await client.lifeguardLogin(phoneValue);
      const token = result.session_token;
      if (!token) {
        throw new Error("Login failed — no session token returned");
      }

      // Store token and fetch full profile (including avatar URLs)
      sessionTokenRef.current = token;
      setSessionToken(token);
      const authClient = createApi(activeUrl, () => token);
      let profile;
      try {
        const me = await authClient.lifeguardMe();
        profile = me || {
          id: result.id,
          name: result.name,
          phone: result.phone,
          zones: result.zones || [],
          online: result.online,
          last_seen: result.last_seen,
        };
      } catch (err) {
        // If profile fetch fails, fallback to login response and keep the session.
        profile = {
          id: result.id,
          name: result.name,
          phone: result.phone,
          zones: result.zones || [],
          online: result.online,
          last_seen: result.last_seen,
        };
      }

      setLifeguard(profile);
      await persistSession(token, profile);
      setConnected(true);
      logInfo("Lifeguard signed in", { name: profile.name, zones: profile.zones });
      return profile;
    },
    [baseUrl, persistSession, saveBaseUrl]
  );

  const logout = useCallback(async () => {
    try {
      if (sessionTokenRef.current) {
        await api.lifeguardLogout();
      }
    } catch {
      // Local logout still proceeds if backend is unreachable.
    } finally {
      sessionTokenRef.current = "";
      setSessionToken("");
      setLifeguard(null);
      await clearSessionStorage();
      logInfo("Lifeguard signed out");
    }
  }, [api, clearSessionStorage]);

  const refreshLifeguard = useCallback(async () => {
    if (!sessionTokenRef.current) return null;
    const me = await api.lifeguardMe();
    setLifeguard(me);
    await persistSession(sessionTokenRef.current, me);
    return me;
  }, [api, persistSession]);

  const clearConnection = useCallback(() => {
    setConnected(false);
    setHealth(null);
  }, []);

  const value = useMemo(
    () => ({
      baseUrl,
      api,
      ready,
      authReady,
      connected,
      health,
      sessionToken,
      lifeguard,
      assignedZones,
      streamQuality,
      saveStreamQuality,
      isAuthenticated: Boolean(sessionToken && lifeguard),
      saveBaseUrl,
      testConnection,
      login,
      logout,
      refreshLifeguard,
      clearConnection,
      setConnected,
      setHealth,
    }),
    [
      baseUrl,
      api,
      ready,
      authReady,
      connected,
      health,
      sessionToken,
      lifeguard,
      assignedZones,
      streamQuality,
      saveStreamQuality,
      saveBaseUrl,
      testConnection,
      login,
      logout,
      refreshLifeguard,
      clearConnection,
    ]
  );

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
}

export function useApiContext() {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error("useApiContext must be used within ApiProvider");
  }
  return context;
}

