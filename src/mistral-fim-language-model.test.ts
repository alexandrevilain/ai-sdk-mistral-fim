import { LanguageModelV2Prompt } from '@ai-sdk/provider';
import {
  convertReadableStreamToArray,
  createTestServer,
  mockId,
} from '@ai-sdk/provider-utils/test';
import { createMistralFim } from './mistral-provider';

const TEST_PROMPT: LanguageModelV2Prompt = [
  { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
];

const provider = createMistralFim({
  apiKey: 'test-api-key',
  generateId: mockId(),
});
const model = provider.chat('mistral-small-latest');

const server = createTestServer({
  'https://api.mistral.ai/v1/fim/completions': {},
});

describe('doGenerate', () => {
  function prepareJsonResponse({
    content = '',
    usage = {
      prompt_tokens: 4,
      total_tokens: 34,
      completion_tokens: 30,
    },
    id = '16362f24e60340d0994dd205c267a43a',
    created = 1711113008,
    model = 'mistral-small-latest',
    headers,
  }: {
    content?: string;
    usage?: {
      prompt_tokens: number;
      total_tokens: number;
      completion_tokens: number;
    };
    id?: string;
    created?: number;
    model?: string;
    headers?: Record<string, string>;
  }) {
    server.urls['https://api.mistral.ai/v1/fim/completions'].response = {
      type: 'json-value',
      headers,
      body: {
        object: 'chat.completion',
        id,
        created,
        model,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content,
              tool_calls: null,
            },
            finish_reason: 'stop',
            logprobs: null,
          },
        ],
        usage,
      },
    };
  }

  it('should extract text content', async () => {
    prepareJsonResponse({ content: 'Hello, World!' });

    const { content } = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(content).toMatchInlineSnapshot(`
      [
        {
          "text": "Hello, World!",
          "type": "text",
        },
      ]
    `);
  });

  it('should avoid duplication when there is a trailing assistant message', async () => {
    prepareJsonResponse({ content: 'prefix and more content' });

    const { content } = await model.doGenerate({
      prompt: [
        { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
        {
          role: 'assistant',
          content: [{ type: 'text', text: 'prefix ' }],
        },
      ],
    });

    expect(content).toMatchInlineSnapshot(`
      [
        {
          "text": "prefix and more content",
          "type": "text",
        },
      ]
    `);
  });

  it('should extract usage', async () => {
    prepareJsonResponse({
      usage: { prompt_tokens: 20, total_tokens: 25, completion_tokens: 5 },
    });

    const { usage } = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(usage).toMatchInlineSnapshot(`
      {
        "inputTokens": 20,
        "outputTokens": 5,
        "totalTokens": 25,
      }
    `);
  });

  it('should send additional response information', async () => {
    prepareJsonResponse({
      id: 'test-id',
      created: 123,
      model: 'test-model',
    });

    const { response } = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect({
      id: response?.id,
      timestamp: response?.timestamp,
      modelId: response?.modelId,
    }).toStrictEqual({
      id: 'test-id',
      timestamp: new Date(123 * 1000),
      modelId: 'test-model',
    });
  });

  it('should expose the raw response headers', async () => {
    prepareJsonResponse({
      headers: { 'test-header': 'test-value' },
    });

    const { response } = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(response?.headers).toStrictEqual({
      // default headers:
      'content-length': '314',
      'content-type': 'application/json',

      // custom header
      'test-header': 'test-value',
    });
  });

  it('should pass the model and the prompt', async () => {
    prepareJsonResponse({ content: '' });

    await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: 'mistral-small-latest',
      prompt: 'Hello',
    });
  });

  it('should pass headers', async () => {
    prepareJsonResponse({ content: '' });

    const provider = createMistralFim({
      apiKey: 'test-api-key',
      headers: {
        'Custom-Provider-Header': 'provider-header-value',
      },
    });

    await provider.chat('mistral-small-latest').doGenerate({
      prompt: TEST_PROMPT,
      headers: {
        'Custom-Request-Header': 'request-header-value',
      },
    });

    const requestHeaders = server.calls[0].requestHeaders;

    expect(requestHeaders).toStrictEqual({
      authorization: 'Bearer test-api-key',
      'content-type': 'application/json',
      'custom-provider-header': 'provider-header-value',
      'custom-request-header': 'request-header-value',
    });
  });

  it('should send request body', async () => {
    prepareJsonResponse({ content: '' });

    const { request } = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(request).toMatchInlineSnapshot(`
      {
        "body": {
          "max_tokens": undefined,
          "min_tokens": undefined,
          "model": "mistral-small-latest",
          "prompt": "Hello",
          "random_seed": undefined,
          "stop": undefined,
          "suffix": undefined,
          "temperature": undefined,
          "top_p": undefined,
        },
      }
    `);
  });

  it('should extract content when message content is a content object', async () => {
    server.urls['https://api.mistral.ai/v1/fim/completions'].response = {
      type: 'json-value',
      body: {
        object: 'chat.completion',
        id: 'object-id',
        created: 1711113008,
        model: 'mistral-small-latest',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: [
                {
                  type: 'text',
                  text: 'Hello from object',
                },
              ],
              tool_calls: null,
            },
            finish_reason: 'stop',
            logprobs: null,
          },
        ],
        usage: { prompt_tokens: 4, total_tokens: 34, completion_tokens: 30 },
      },
    };

    const { content } = await model.doGenerate({
      prompt: TEST_PROMPT,
    });

    expect(content).toMatchInlineSnapshot(`
      [
        {
          "text": "Hello from object",
          "type": "text",
        },
      ]
    `);
  });
});

describe('doStream', () => {
  function prepareStreamResponse({
    content,
    headers,
  }: {
    content: string[];
    headers?: Record<string, string>;
  }) {
    server.urls['https://api.mistral.ai/v1/fim/completions'].response = {
      type: 'stream-chunks',
      headers,
      chunks: [
        `data:  {"id":"53ff663126294946a6b7a4747b70597e","object":"chat.completion.chunk",` +
          `"created":1750537996,"model":"mistral-small-latest","choices":[{"index":0,` +
          `"delta":{"role":"assistant","content":""},"finish_reason":null,"logprobs":null}]}\n\n`,
        ...content.map((text) => {
          return (
            `data:  {"id":"53ff663126294946a6b7a4747b70597e","object":"chat.completion.chunk",` +
            `"created":1750537996,"model":"mistral-small-latest","choices":[{"index":0,` +
            `"delta":{"role":"assistant","content":"${text}"},"finish_reason":null,"logprobs":null}]}\n\n`
          );
        }),
        `data:  {"id":"53ff663126294946a6b7a4747b70597e","object":"chat.completion.chunk",` +
          `"created":1750537996,"model":"mistral-small-latest","choices":[{"index":0,` +
          `"delta":{"content":""},"finish_reason":"stop","logprobs":null}],` +
          `"usage":{"prompt_tokens":4,"total_tokens":36,"completion_tokens":32}}\n\n`,
        `data: [DONE]\n\n`,
      ],
    };
  }

  it('should stream text deltas', async () => {
    prepareStreamResponse({ content: ['Hello', ', ', 'world!'] });

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      includeRawChunks: false,
    });

    expect(await convertReadableStreamToArray(stream)).toMatchInlineSnapshot(`
      [
        {
          "type": "stream-start",
          "warnings": [],
        },
        {
          "id": "53ff663126294946a6b7a4747b70597e",
          "modelId": "mistral-small-latest",
          "timestamp": 2025-06-21T20:33:16.000Z,
          "type": "response-metadata",
        },
        {
          "id": "0",
          "type": "text-start",
        },
        {
          "delta": "Hello",
          "id": "0",
          "type": "text-delta",
        },
        {
          "delta": ", ",
          "id": "0",
          "type": "text-delta",
        },
        {
          "delta": "world!",
          "id": "0",
          "type": "text-delta",
        },
        {
          "id": "0",
          "type": "text-end",
        },
        {
          "finishReason": "stop",
          "type": "finish",
          "usage": {
            "inputTokens": 4,
            "outputTokens": 32,
            "totalTokens": 36,
          },
        },
      ]
    `);
  });

  it('should avoid duplication when there is a trailing assistant message', async () => {
    prepareStreamResponse({ content: ['prefix', ' and', ' more content'] });

    const { stream } = await model.doStream({
      prompt: [
        { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
        {
          role: 'assistant',
          content: [{ type: 'text', text: 'prefix ' }],
        },
      ],
      includeRawChunks: false,
    });

    expect(await convertReadableStreamToArray(stream)).toMatchInlineSnapshot(`
      [
        {
          "type": "stream-start",
          "warnings": [],
        },
        {
          "id": "53ff663126294946a6b7a4747b70597e",
          "modelId": "mistral-small-latest",
          "timestamp": 2025-06-21T20:33:16.000Z,
          "type": "response-metadata",
        },
        {
          "id": "0",
          "type": "text-start",
        },
        {
          "delta": "prefix",
          "id": "0",
          "type": "text-delta",
        },
        {
          "delta": " and",
          "id": "0",
          "type": "text-delta",
        },
        {
          "delta": " more content",
          "id": "0",
          "type": "text-delta",
        },
        {
          "id": "0",
          "type": "text-end",
        },
        {
          "finishReason": "stop",
          "type": "finish",
          "usage": {
            "inputTokens": 4,
            "outputTokens": 32,
            "totalTokens": 36,
          },
        },
      ]
    `);
  });

  it('should expose the raw response headers', async () => {
    prepareStreamResponse({
      content: [],
      headers: { 'test-header': 'test-value' },
    });

    const { response } = await model.doStream({
      prompt: TEST_PROMPT,
      includeRawChunks: false,
    });

    expect(response?.headers).toStrictEqual({
      // default headers:
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',

      // custom header
      'test-header': 'test-value',
    });
  });

  it('should pass the prompt', async () => {
    prepareStreamResponse({ content: [''] });

    await model.doStream({
      prompt: TEST_PROMPT,
      includeRawChunks: false,
    });

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      stream: true,
      model: 'mistral-small-latest',
      prompt: 'Hello',
    });
  });

  it('should pass headers', async () => {
    prepareStreamResponse({ content: [] });

    const provider = createMistralFim({
      apiKey: 'test-api-key',
      headers: {
        'Custom-Provider-Header': 'provider-header-value',
      },
    });

    await provider.chat('mistral-small-latest').doStream({
      prompt: TEST_PROMPT,
      includeRawChunks: false,
      headers: {
        'Custom-Request-Header': 'request-header-value',
      },
    });

    expect(server.calls[0].requestHeaders).toStrictEqual({
      authorization: 'Bearer test-api-key',
      'content-type': 'application/json',
      'custom-provider-header': 'provider-header-value',
      'custom-request-header': 'request-header-value',
    });
  });

  it('should send request body', async () => {
    prepareStreamResponse({ content: [] });

    const { request } = await model.doStream({
      prompt: TEST_PROMPT,
      includeRawChunks: false,
    });

    expect(request).toMatchInlineSnapshot(`
      {
        "body": {
          "max_tokens": undefined,
          "min_tokens": undefined,
          "model": "mistral-small-latest",
          "prompt": "Hello",
          "random_seed": undefined,
          "stop": undefined,
          "stream": true,
          "suffix": undefined,
          "temperature": undefined,
          "top_p": undefined,
        },
      }
    `);
  });

  it('should stream text with content objects', async () => {
    // Instead of using prepareStreamResponse (which sends strings),
    // we set the chunks manually so that each delta's content is an object.
    server.urls['https://api.mistral.ai/v1/fim/completions'].response = {
      type: 'stream-chunks',
      chunks: [
        `data: {"id":"b9e43f82d6c74a1e9f5b2c8e7a9d4f6b","object":"chat.completion.chunk","created":1750538500,"model":"mistral-small-latest","choices":[{"index":0,"delta":{"role":"assistant","content":[{"type":"text","text":""}]},"finish_reason":null,"logprobs":null}]}\n\n`,
        `data: {"id":"b9e43f82d6c74a1e9f5b2c8e7a9d4f6b","object":"chat.completion.chunk","created":1750538500,"model":"mistral-small-latest","choices":[{"index":0,"delta":{"content":[{"type":"text","text":"Hello"}]},"finish_reason":null,"logprobs":null}]}\n\n`,
        `data: {"id":"b9e43f82d6c74a1e9f5b2c8e7a9d4f6b","object":"chat.completion.chunk","created":1750538500,"model":"mistral-small-latest","choices":[{"index":0,"delta":{"content":[{"type":"text","text":", world!"}]},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":4,"total_tokens":36,"completion_tokens":32}}\n\n`,
        `data: [DONE]\n\n`,
      ],
    };

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      includeRawChunks: false,
    });

    expect(await convertReadableStreamToArray(stream)).toMatchInlineSnapshot(`
      [
        {
          "type": "stream-start",
          "warnings": [],
        },
        {
          "id": "b9e43f82d6c74a1e9f5b2c8e7a9d4f6b",
          "modelId": "mistral-small-latest",
          "timestamp": 2025-06-21T20:41:40.000Z,
          "type": "response-metadata",
        },
        {
          "id": "0",
          "type": "text-start",
        },
        {
          "delta": "Hello",
          "id": "0",
          "type": "text-delta",
        },
        {
          "delta": ", world!",
          "id": "0",
          "type": "text-delta",
        },
        {
          "id": "0",
          "type": "text-end",
        },
        {
          "finishReason": "stop",
          "type": "finish",
          "usage": {
            "inputTokens": 4,
            "outputTokens": 32,
            "totalTokens": 36,
          },
        },
      ]
    `);
  });
});

describe('doStream with raw chunks', () => {
  it('should stream raw chunks when includeRawChunks is true', async () => {
    server.urls['https://api.mistral.ai/v1/fim/completions'].response = {
      type: 'stream-chunks',
      chunks: [
        `data: {"id":"c7d54e93f8a64b2e9c1f5a8b7d9e2f4c","object":"chat.completion.chunk","created":1750538600,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null,"logprobs":null}]}\n\n`,
        `data: {"id":"d8e65fa4g9b75c3f0d2g6b9c8e0f3g5d","object":"chat.completion.chunk","created":1750538601,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null,"logprobs":null}]}\n\n`,
        `data: {"id":"e9f76gb5h0c86d4g1e3h7c0d9f1g4h6e","object":"chat.completion.chunk","created":1750538602,"model":"mistral-large-latest","choices":[{"index":0,"delta":{},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n`,
        'data: [DONE]\n\n',
      ],
    };

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      includeRawChunks: true,
    });

    expect(await convertReadableStreamToArray(stream)).toMatchInlineSnapshot(`
      [
        {
          "type": "stream-start",
          "warnings": [],
        },
        {
          "rawValue": {
            "choices": [
              {
                "delta": {
                  "content": "Hello",
                  "role": "assistant",
                },
                "finish_reason": null,
                "index": 0,
                "logprobs": null,
              },
            ],
            "created": 1750538600,
            "id": "c7d54e93f8a64b2e9c1f5a8b7d9e2f4c",
            "model": "mistral-large-latest",
            "object": "chat.completion.chunk",
          },
          "type": "raw",
        },
        {
          "id": "c7d54e93f8a64b2e9c1f5a8b7d9e2f4c",
          "modelId": "mistral-large-latest",
          "timestamp": 2025-06-21T20:43:20.000Z,
          "type": "response-metadata",
        },
        {
          "id": "0",
          "type": "text-start",
        },
        {
          "delta": "Hello",
          "id": "0",
          "type": "text-delta",
        },
        {
          "rawValue": {
            "choices": [
              {
                "delta": {
                  "content": " world",
                },
                "finish_reason": null,
                "index": 0,
                "logprobs": null,
              },
            ],
            "created": 1750538601,
            "id": "d8e65fa4g9b75c3f0d2g6b9c8e0f3g5d",
            "model": "mistral-large-latest",
            "object": "chat.completion.chunk",
          },
          "type": "raw",
        },
        {
          "delta": " world",
          "id": "0",
          "type": "text-delta",
        },
        {
          "rawValue": {
            "choices": [
              {
                "delta": {},
                "finish_reason": "stop",
                "index": 0,
                "logprobs": null,
              },
            ],
            "created": 1750538602,
            "id": "e9f76gb5h0c86d4g1e3h7c0d9f1g4h6e",
            "model": "mistral-large-latest",
            "object": "chat.completion.chunk",
            "usage": {
              "completion_tokens": 5,
              "prompt_tokens": 10,
              "total_tokens": 15,
            },
          },
          "type": "raw",
        },
        {
          "id": "0",
          "type": "text-end",
        },
        {
          "finishReason": "stop",
          "type": "finish",
          "usage": {
            "inputTokens": 10,
            "outputTokens": 5,
            "totalTokens": 15,
          },
        },
      ]
    `);
  });
});
