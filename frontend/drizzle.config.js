/** @type { import("drizzle-kit").Config } */
const { defineConfig } = require('drizzle-kit');
export default defineConfig({
  dialect: "postgresql", // "mysql" | "sqlite" | "postgresql"
  schema: "./utils/schema.js",
  dbCredentials:{
    url: process.env.NEXT_PUBLIC_DRIZZLE_DB_URL,
  }
});
