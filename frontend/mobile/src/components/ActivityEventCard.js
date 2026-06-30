import React, { memo } from "react";
import { StyleSheet, View } from "react-native";
import { Button, Card, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { formatTimestamp } from "../utils/format";
import { getEventSeverityColor } from "../utils/activityEvents";
import { colors, spacing } from "../theme";

function ActivityEventCard({ event, onRespond, responding = false, responded = false }) {
  const accent = getEventSeverityColor(event, colors);
  const showRespond = event.respondable && onRespond && !responded;

  return (
    <Card style={[styles.card, { borderLeftColor: accent }]}>
      <View style={styles.row}>
        <View style={[styles.iconWrap, { backgroundColor: `${accent}22` }]}>
          <MaterialCommunityIcons name={event.icon || "information"} size={22} color={accent} />
        </View>
        <View style={styles.content}>
          <Text variant="titleSmall" style={[styles.title, { color: accent }]}>
            {event.title}
          </Text>
          <Text style={styles.meta}>Zone {event.zone}</Text>
          {event.subtitle ? <Text style={styles.meta}>{event.subtitle}</Text> : null}
          <Text style={styles.time}>{formatTimestamp(event.timestamp)}</Text>
          {showRespond ? (
            <Button
              mode="contained-tonal"
              compact
              icon="shield-check"
              onPress={() => onRespond(event)}
              loading={responding}
              disabled={responding}
              style={styles.respondBtn}
              buttonColor={`${accent}33`}
              textColor={accent}
            >
              Respond
            </Button>
          ) : responded ? (
            <Text style={[styles.responded, { color: colors.success }]}>Response recorded</Text>
          ) : null}
        </View>
      </View>
    </Card>
  );
}

export default memo(ActivityEventCard);

const styles = StyleSheet.create({
  card: {
    backgroundColor: "rgba(16,34,53,0.68)",
    marginBottom: spacing.md,
    borderLeftWidth: 4,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.22)",
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  row: {
    flexDirection: "row",
    padding: spacing.md,
    gap: spacing.md,
    alignItems: "flex-start",
  },
  iconWrap: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1.5,
  },
  content: {
    flex: 1,
  },
  title: {
    fontWeight: "900",
    marginBottom: spacing.sm,
    fontSize: 15.5,
    letterSpacing: 0.15,
  },
  meta: {
    color: colors.textMuted,
    fontSize: 13,
    marginBottom: 3,
    fontWeight: "600",
  },
  time: {
    color: colors.textMuted,
    fontSize: 12,
    marginTop: spacing.xs,
    fontWeight: "600",
  },
  respondBtn: {
    alignSelf: "flex-start",
    marginTop: spacing.md,
    borderRadius: 10,
  },
  responded: {
    fontSize: 13,
    fontWeight: "800",
    marginTop: spacing.md,
    letterSpacing: 0.1,
  },
});
