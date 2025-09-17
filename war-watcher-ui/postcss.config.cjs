// CommonJS here because many PostCSS tools still assume CJS.
// If you keep this file as .js while using "type":"module", it will throw.
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
