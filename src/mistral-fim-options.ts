import { z } from 'zod/v4';

export type MistralFimModelId =
  | 'codestral-2405'
  | 'codestral-latest'
  | (string & {});

export const mistralFimProviderOptions = z.object({
  stop: z.union([z.string(), z.array(z.string())]).optional(),
  suffix: z.string().optional(),
  min_token: z.number().optional(),
});

export type MistralFimProviderOptions = z.infer<
  typeof mistralFimProviderOptions
>;
