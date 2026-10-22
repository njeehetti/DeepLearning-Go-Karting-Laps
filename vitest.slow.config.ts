import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: { include: ['tests/**/*.slow.test.ts'], testTimeout: 300_000 },
});
