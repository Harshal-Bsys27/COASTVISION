/** Empty assignedZones = lifeguard sees all zones (admin/default). */
export function hasZoneRestriction(assignedZones) {
  return Array.isArray(assignedZones) && assignedZones.length > 0;
}

export function isZoneAllowed(zoneId, assignedZones) {
  if (!hasZoneRestriction(assignedZones)) return true;
  const id = Number(zoneId);
  return assignedZones.some((z) => Number(z) === id);
}

export function filterZones(zones, assignedZones) {
  if (!Array.isArray(zones)) return [];
  if (!hasZoneRestriction(assignedZones)) return zones;
  return zones.filter((zone) => isZoneAllowed(zone.id ?? zone.zone, assignedZones));
}

export function filterAlerts(alerts, assignedZones) {
  if (!Array.isArray(alerts)) return [];
  if (!hasZoneRestriction(assignedZones)) return alerts;
  return alerts.filter((alert) => isZoneAllowed(alert.zone, assignedZones));
}

export function filterAnalysis(analysis, assignedZones) {
  if (!analysis || !hasZoneRestriction(assignedZones)) return analysis;

  const allowed = new Set(assignedZones.map((z) => String(z)));
  const alertsByZone = {};
  let alertsTotal = 0;

  for (const [zone, count] of Object.entries(analysis.alerts_by_zone || {})) {
    if (allowed.has(String(zone))) {
      alertsByZone[zone] = count;
      alertsTotal += count;
    }
  }

  return {
    ...analysis,
    alerts_total: alertsTotal,
    alerts_by_zone: alertsByZone,
  };
}

export function filterCrowdZones(crowdZones, assignedZones) {
  if (!Array.isArray(crowdZones)) return [];
  if (!hasZoneRestriction(assignedZones)) return crowdZones;
  return crowdZones.filter((zone) => isZoneAllowed(zone.zone ?? zone.id, assignedZones));
}

export function filterByZoneField(items, assignedZones, getZoneId = (item) => item.zone) {
  if (!Array.isArray(items)) return [];
  if (!hasZoneRestriction(assignedZones)) return items;
  return items.filter((item) => isZoneAllowed(getZoneId(item), assignedZones));
}

