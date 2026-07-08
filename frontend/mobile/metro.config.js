const { getDefaultConfig } = (() => {
  try {
    return require("expo/metro-config");
  } catch {
    return require("@expo/metro-config");
  }
})();

const config = getDefaultConfig(__dirname);
config.resolver = config.resolver || {};
config.resolver.blockList = [/.*\.claude[\\/].*$/];

module.exports = config;

