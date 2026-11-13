// SPDX-License-Identifier: GPL-3.0-only
import { vocabulary } from './encoders/v21.ts';
process.stdout.write(JSON.stringify({ size: vocabulary.names.length, sha256: vocabulary.sha256 }) + '\n');
