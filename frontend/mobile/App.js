import React, { useEffect } from "react";
import { Platform, StatusBar as RNStatusBar } from "react-native";
import { StatusBar } from "expo-status-bar";
import { Provider as PaperProvider, MD3DarkTheme } from "react-native-paper";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { ApiProvider } from "./src/context/ApiContext";
import RootNavigator from "./src/navigation/RootNavigator";
import { logInfo } from "./src/utils/logger";
import { colors } from "./src/theme";

const paperTheme = {
  ...MD3DarkTheme,
  colors: {
    ...MD3DarkTheme.colors,
    primary: colors.primary,
    background: colors.background,
    surface: colors.surface,
    onSurface: colors.text,
  },
};

export default function App() {
  useEffect(() => {
    if (Platform.OS === "android") {
      RNStatusBar.setTranslucent(false);
      RNStatusBar.setBackgroundColor(colors.background);
      RNStatusBar.setBarStyle("light-content");
    }
    logInfo("App started — logs appear in this terminal when using Expo Go");
  }, []);

  return (
    <SafeAreaProvider>
      <PaperProvider theme={paperTheme}>
        <ApiProvider>
          <StatusBar style="light" />
          <RootNavigator />
        </ApiProvider>
      </PaperProvider>
    </SafeAreaProvider>
  );
}
