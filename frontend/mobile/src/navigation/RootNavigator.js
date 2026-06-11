import React from "react";
import { ActivityIndicator, View } from "react-native";
import { NavigationContainer, DarkTheme } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import LifeguardSessionEffects from "../components/LifeguardSessionEffects";
import ScreenContainer from "../components/ScreenContainer";
import { useApiContext } from "../context/ApiContext";
import DashboardScreen from "../screens/DashboardScreen";
import AnalyticsScreen from "../screens/AnalyticsScreen";
import EventLogsScreen from "../screens/EventLogsScreen";
import SettingsScreen from "../screens/SettingsScreen";
import SignInScreen from "../screens/SignInScreen";
import ZoneDetailScreen from "../screens/ZoneDetailScreen";
import { colors, layout } from "../theme";

function withScreenSafeArea(Component) {
  function SafeAreaScreen(props) {
    return (
      <ScreenContainer>
        <Component {...props} />
      </ScreenContainer>
    );
  }
  SafeAreaScreen.displayName = `SafeArea(${Component.displayName || Component.name || "Screen"})`;
  return SafeAreaScreen;
}

const SafeAnalyticsScreen = withScreenSafeArea(AnalyticsScreen);
const SafeEventLogsScreen = withScreenSafeArea(EventLogsScreen);
const SafeSettingsScreen = withScreenSafeArea(SettingsScreen);

const Tab = createBottomTabNavigator();
const DashboardStack = createNativeStackNavigator();
const AuthStack = createNativeStackNavigator();

const navTheme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    background: colors.background,
    card: colors.surface,
    text: colors.text,
    border: colors.border,
    primary: colors.primary,
  },
};

function DashboardStackScreen() {
  const insets = useSafeAreaInsets();

  return (
    <DashboardStack.Navigator
      screenOptions={{
        headerStyle: {
          backgroundColor: colors.surface,
        },
        headerTintColor: colors.text,
        contentStyle: { backgroundColor: colors.background },
        statusBarStyle: "light",
        statusBarColor: colors.background,
        headerStatusBarHeight: insets.top,
      }}
    >
      <DashboardStack.Screen name="DashboardHome" component={DashboardScreen} options={{ title: "Dashboard" }} />
      <DashboardStack.Screen name="ZoneDetail" component={ZoneDetailScreen} options={{ title: "Zone Detail" }} />
    </DashboardStack.Navigator>
  );
}

function MainTabNavigator() {
  const insets = useSafeAreaInsets();
  const tabBarHeight = layout.bottomTabHeight + insets.bottom;

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: {
          backgroundColor: colors.surface,
          borderTopColor: colors.border,
          height: tabBarHeight,
          paddingBottom: Math.max(insets.bottom, 8),
          paddingTop: 6,
        },
        tabBarActiveTintColor: colors.primary,
        tabBarInactiveTintColor: colors.textMuted,
        tabBarIcon: ({ color, size }) => {
            const icons = {
              Dashboard: "view-dashboard",
              Analytics: "chart-line",
              Logs: "clipboard-text-clock",
              Settings: "cog",
            };
          return <MaterialCommunityIcons name={icons[route.name]} size={size} color={color} />;
        },
      })}
    >
      <Tab.Screen name="Dashboard" component={DashboardStackScreen} />
      <Tab.Screen name="Analytics" component={SafeAnalyticsScreen} />
      <Tab.Screen name="Logs" component={SafeEventLogsScreen} />
      <Tab.Screen name="Settings" component={SafeSettingsScreen} />
    </Tab.Navigator>
  );
}

function AuthNavigator() {
  return (
    <AuthStack.Navigator screenOptions={{ headerShown: false, contentStyle: { backgroundColor: colors.background } }}>
      <AuthStack.Screen name="SignIn" component={SignInScreen} />
    </AuthStack.Navigator>
  );
}

function LoadingScreen() {
  return (
    <View style={{ flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: colors.background }}>
      <ActivityIndicator color={colors.primary} size="large" />
    </View>
  );
}

export default function RootNavigator() {
  const { authReady, isAuthenticated } = useApiContext();

  if (!authReady) {
    return <LoadingScreen />;
  }

  return (
    <NavigationContainer theme={navTheme}>
      {isAuthenticated ? (
        <>
          <LifeguardSessionEffects />
          <MainTabNavigator />
        </>
      ) : (
        <AuthNavigator />
      )}
    </NavigationContainer>
  );
}
