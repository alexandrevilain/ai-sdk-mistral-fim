# AI SDK - Mistral FIM (fill-in-the-middle) Provider

The Mistral fill-in-the-middle provider for the [AI SDK](https://ai-sdk.dev/docs) contains language model support for the Mistral FIM API.

This project is a fork of the official provider with the following changes:
- Uses the FIM (fill-in-the-middle) endpoint instead of chat endpoint.
- Remove tools calls and reasoning support (as they are not supported in FIM endpoint)

This provider has been created for the VSCode extension [TabCoder](https://github.com/alexandrevilain/tabcoder).

## Setup

The Mistral FIM provider is available in the `ai-sdk-mistral-fim` module. You can install it with:

```bash
npm i ai-sdk-mistral-fim
```

## Provider Instance

You can import the default provider instance `mistralFim` from `ai-sdk-mistral-fim`:

```ts
import { mistralFim } from 'ai-sdk-mistral-fim';
```

## Example

```ts
import { mistralFim } from 'ai-sdk-mistral-fim';
import { generateText } from 'ai';

const { text } = await generateText({
  model: mistralFim('mistral-large-latest'),
  prompt: 'Write a vegetarian lasagna recipe for 4 people.',
});
```

## Documentation
