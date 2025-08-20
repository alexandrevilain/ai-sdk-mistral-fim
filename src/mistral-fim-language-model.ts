import {
  LanguageModelV2,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  LanguageModelV2FinishReason,
  LanguageModelV2Prompt,
  LanguageModelV2StreamPart,
  LanguageModelV2Usage,
} from '@ai-sdk/provider';
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
  FetchFunction,
  generateId,
  parseProviderOptions,
  ParseResult,
  postJsonToApi,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';
import { getResponseMetadata } from './get-response-metadata';
import { mapMistralFinishReason } from './map-mistral-finish-reason';
import {
  MistralFimModelId,
  mistralFimProviderOptions,
} from './mistral-fim-options';
import { mistralFailedResponseHandler } from './mistral-error';

type MistralChatConfig = {
  provider: string;
  baseURL: string;
  headers: () => Record<string, string | undefined>;
  fetch?: FetchFunction;
  generateId?: () => string;
};

export class MistralFimLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = 'v2';

  readonly modelId: MistralFimModelId;

  private readonly config: MistralChatConfig;
  private readonly generateId: () => string;

  constructor(modelId: MistralFimModelId, config: MistralChatConfig) {
    this.modelId = modelId;
    this.config = config;
    this.generateId = config.generateId ?? generateId;
  }

  get provider(): string {
    return this.config.provider;
  }

  readonly supportedUrls: Record<string, RegExp[]> = {};

  private async getArgs({
    prompt,
    maxOutputTokens,
    temperature,
    topP,
    topK,
    frequencyPenalty,
    presencePenalty,
    stopSequences,
    responseFormat,
    seed,
    providerOptions,
    tools,
    toolChoice,
  }: Parameters<LanguageModelV2['doGenerate']>[0]) {
    const warnings: LanguageModelV2CallWarning[] = [];

    const options =
      (await parseProviderOptions({
        provider: this.provider,
        providerOptions,
        schema: mistralFimProviderOptions,
      })) ?? {};

    if (topK != null) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'topK',
      });
    }

    if (frequencyPenalty != null) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'frequencyPenalty',
      });
    }

    if (presencePenalty != null) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'presencePenalty',
      });
    }

    if (stopSequences != null) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'stopSequences',
      });
    }

    if (responseFormat != null) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'responseFormat',
      });
    }

    if (tools != null) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'tools',
      });
    }

    if (toolChoice != null) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'toolChoice',
      });
    }

    let prefix = extractUserPromptContent(prompt);
    if (!prefix) {
      throw new Error('Empty prompt');
    }

    const args = {
      model: this.modelId,
      temperature,
      top_p: topP,
      max_tokens: maxOutputTokens,
      stop: options.stop,
      random_seed: seed,
      prompt: prefix,
      suffix: options.suffix,
      min_tokens: options.min_token,
    };

    return {
      args,
      warnings,
    };
  }

  async doGenerate(
    options: Parameters<LanguageModelV2['doGenerate']>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2['doGenerate']>>> {
    const { args: body, warnings } = await this.getArgs(options);

    const {
      responseHeaders,
      value: response,
      rawValue: rawResponse,
    } = await postJsonToApi({
      url: `${this.config.baseURL}/fim/completions`,
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: mistralFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        mistralChatResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const choice = response.choices[0];
    const content: Array<LanguageModelV2Content> = [];

    if (
      choice.message.content != null &&
      Array.isArray(choice.message.content)
    ) {
      for (const part of choice.message.content) {
        if (part.type === 'text') {
          if (part.text.length > 0) {
            content.push({ type: 'text', text: part.text });
          }
        }
      }
    } else {
      const text = extractTextContent(choice.message.content);
      if (text != null && text.length > 0) {
        content.push({ type: 'text', text });
      }
    }

    return {
      content,
      finishReason: mapMistralFinishReason(choice.finish_reason),
      usage: {
        inputTokens: response.usage.prompt_tokens,
        outputTokens: response.usage.completion_tokens,
        totalTokens: response.usage.total_tokens,
      },
      request: { body },
      response: {
        ...getResponseMetadata(response),
        headers: responseHeaders,
        body: rawResponse,
      },
      warnings,
    };
  }

  async doStream(
    options: Parameters<LanguageModelV2['doStream']>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2['doStream']>>> {
    const { args, warnings } = await this.getArgs(options);
    const body = { ...args, stream: true };

    const { responseHeaders, value: response } = await postJsonToApi({
      url: `${this.config.baseURL}/fim/completions`,
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: mistralFailedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(
        mistralChatChunkSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    let finishReason: LanguageModelV2FinishReason = 'unknown';
    const usage: LanguageModelV2Usage = {
      inputTokens: undefined,
      outputTokens: undefined,
      totalTokens: undefined,
    };

    let isFirstChunk = true;
    let activeText = false;

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof mistralChatChunkSchema>>,
          LanguageModelV2StreamPart
        >({
          start(controller) {
            controller.enqueue({ type: 'stream-start', warnings });
          },

          transform(chunk, controller) {
            // Emit raw chunk if requested (before anything else)
            if (options.includeRawChunks) {
              controller.enqueue({ type: 'raw', rawValue: chunk.rawValue });
            }

            if (!chunk.success) {
              controller.enqueue({ type: 'error', error: chunk.error });
              return;
            }

            const value = chunk.value;

            if (isFirstChunk) {
              isFirstChunk = false;

              controller.enqueue({
                type: 'response-metadata',
                ...getResponseMetadata(value),
              });
            }

            if (value.usage != null) {
              usage.inputTokens = value.usage.prompt_tokens;
              usage.outputTokens = value.usage.completion_tokens;
              usage.totalTokens = value.usage.total_tokens;
            }

            const choice = value.choices[0];
            const delta = choice.delta;

            const textContent = extractTextContent(delta.content);

            if (textContent != null && textContent.length > 0) {
              if (!activeText) {
                controller.enqueue({ type: 'text-start', id: '0' });
                activeText = true;
              }

              controller.enqueue({
                type: 'text-delta',
                id: '0',
                delta: textContent,
              });
            }

            if (choice.finish_reason != null) {
              finishReason = mapMistralFinishReason(choice.finish_reason);
            }
          },

          flush(controller) {
            if (activeText) {
              controller.enqueue({ type: 'text-end', id: '0' });
            }

            controller.enqueue({
              type: 'finish',
              finishReason,
              usage,
            });
          },
        }),
      ),
      request: { body },
      response: { headers: responseHeaders },
    };
  }
}

function extractTextContent(content: z.infer<typeof mistralContentSchema>) {
  if (typeof content === 'string') {
    return content;
  }

  if (content == null) {
    return undefined;
  }

  const textContent: string[] = [];

  for (const chunk of content) {
    const { type } = chunk;

    switch (type) {
      case 'text':
        textContent.push(chunk.text);
        break;
      case 'thinking':
      case 'image_url':
      case 'reference':
        break;
      default: {
        const _exhaustiveCheck: never = type;
        throw new Error(`Unsupported type: ${_exhaustiveCheck}`);
      }
    }
  }

  return textContent.length ? textContent.join('') : undefined;
}

function extractUserPromptContent(
  prompt: LanguageModelV2Prompt,
): string | undefined {
  for (const message of prompt) {
    if (message.role !== 'user') {
      continue;
    }

    for (const textPart of message.content) {
      if (textPart.type !== 'text') {
        continue;
      }

      return textPart.text;
    }
  }

  return undefined;
}

const mistralContentSchema = z
  .union([
    z.string(),
    z.array(
      z.discriminatedUnion('type', [
        z.object({
          type: z.literal('text'),
          text: z.string(),
        }),
        z.object({
          type: z.literal('image_url'),
          image_url: z.union([
            z.string(),
            z.object({
              url: z.string(),
              detail: z.string().nullable(),
            }),
          ]),
        }),
        z.object({
          type: z.literal('reference'),
          reference_ids: z.array(z.number()),
        }),
        z.object({
          type: z.literal('thinking'),
          thinking: z.array(
            z.object({
              type: z.literal('text'),
              text: z.string(),
            }),
          ),
        }),
      ]),
    ),
  ])
  .nullish();

const mistralUsageSchema = z.object({
  prompt_tokens: z.number(),
  completion_tokens: z.number(),
  total_tokens: z.number(),
});

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const mistralChatResponseSchema = z.object({
  id: z.string().nullish(),
  created: z.number().nullish(),
  model: z.string().nullish(),
  choices: z.array(
    z.object({
      message: z.object({
        role: z.literal('assistant'),
        content: mistralContentSchema,
        tool_calls: z
          .array(
            z.object({
              id: z.string(),
              function: z.object({ name: z.string(), arguments: z.string() }),
            }),
          )
          .nullish(),
      }),
      index: z.number(),
      finish_reason: z.string().nullish(),
    }),
  ),
  object: z.literal('chat.completion'),
  usage: mistralUsageSchema,
});

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const mistralChatChunkSchema = z.object({
  id: z.string().nullish(),
  created: z.number().nullish(),
  model: z.string().nullish(),
  choices: z.array(
    z.object({
      delta: z.object({
        role: z.enum(['assistant']).optional(),
        content: mistralContentSchema,
        tool_calls: z
          .array(
            z.object({
              id: z.string(),
              function: z.object({ name: z.string(), arguments: z.string() }),
            }),
          )
          .nullish(),
      }),
      finish_reason: z.string().nullish(),
      index: z.number(),
    }),
  ),
  usage: mistralUsageSchema.nullish(),
});
