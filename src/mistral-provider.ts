import {
  LanguageModelV2,
  NoSuchModelError,
  ProviderV2,
} from '@ai-sdk/provider';
import {
  FetchFunction,
  loadApiKey,
  withoutTrailingSlash,
} from '@ai-sdk/provider-utils';
import { MistralFimLanguageModel } from './mistral-fim-language-model';
import { MistralFimModelId } from './mistral-fim-options';

export interface MistralFimProvider extends ProviderV2 {
  (modelId: MistralFimModelId): LanguageModelV2;

  /**
Creates a model for text generation.
*/
  languageModel(modelId: MistralFimModelId): LanguageModelV2;

  /**
Creates a model for text generation.
*/
  chat(modelId: MistralFimModelId): LanguageModelV2;
}

export interface MistralProviderSettings {
  /**
Use a different URL prefix for API calls, e.g. to use proxy servers.
The default prefix is `https://api.mistral.ai/v1`.
   */
  baseURL?: string;

  /**
API key that is being send using the `Authorization` header.
It defaults to the `MISTRAL_API_KEY` environment variable.
   */
  apiKey?: string;

  /**
Custom headers to include in the requests.
     */
  headers?: Record<string, string>;

  /**
Custom fetch implementation. You can use it as a middleware to intercept requests,
or to provide a custom fetch implementation for e.g. testing.
    */
  fetch?: FetchFunction;

  generateId?: () => string;
}

/**
Create a Mistral AI provider instance.
 */
export function createMistralFim(
  options: MistralProviderSettings = {},
): MistralFimProvider {
  const baseURL =
    withoutTrailingSlash(options.baseURL) ?? 'https://api.mistral.ai/v1';

  const getHeaders = () => ({
    Authorization: `Bearer ${loadApiKey({
      apiKey: options.apiKey,
      environmentVariableName: 'MISTRAL_API_KEY',
      description: 'Mistral',
    })}`,
    ...options.headers,
  });

  const createFimModel = (modelId: MistralFimModelId) =>
    new MistralFimLanguageModel(modelId, {
      provider: 'mistral.fim',
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
      generateId: options.generateId,
    });

  const provider = function (modelId: MistralFimModelId) {
    if (new.target) {
      throw new Error(
        'The Mistral model function cannot be called with the new keyword.',
      );
    }

    return createFimModel(modelId);
  };

  provider.languageModel = createFimModel;
  provider.chat = createFimModel;

  provider.textEmbeddingModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: 'textEmbeddingModel' });
  };
  provider.imageModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: 'imageModel' });
  };

  return provider;
}

/**
Default Mistral provider instance.
 */
export const mistralFim = createMistralFim();
