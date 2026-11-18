import { defineProject } from "vitest/config";

// Node-side checks of the examples (the demos themselves run in Playwright).
export default defineProject({
  test: {
    name: "examples",
    include: ["test/**/*.test.ts"],
    environment: "node",
  },
});
