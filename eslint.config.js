import js from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'legacy', 'node_modules', 'scripts/*.cjs'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
);
