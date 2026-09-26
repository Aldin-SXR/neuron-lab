# Neuron Lab

A browser-based neural network playground for university beginners. This first release implements feedforward ANNs end to end. CNN, RNN, and LSTM are explicitly marked as planned in the interface.

## Run

Requires Node.js 22.12+ (tested with Node.js 24).

```sh
npm install
npm run dev
```

Open the URL printed by Vite, normally `http://127.0.0.1:5173`.

```sh
npm run build         # Type checking and production build in dist/
npm run preview      # Serve the production build locally
npm test             # Numerical gradient and training tests
npm run test:browser # Chromium user-flow tests; manages its own dev server
```

If Chromium is not installed, run `npx playwright install chromium` before browser tests.

## Explore

- **Learn:** step through each neuron's forward calculation, the sample loss, each neuron's backward gradient, and each individual weight/bias update. Play, pause, change speed, or move backward and forward through recorded states.
- **Train:** train one epoch per step/tick using stochastic gradient descent. Switching from a partially completed lesson finishes its pending updates before training the epoch. Thus the first epoch count may be fractional.
- **Inspect:** select any neuron or connection for formulas with its actual numbers. Blue input neurons, purple hidden neurons, and green output neurons show activations. Positive and negative connections use different colors; thickness represents weight magnitude.
- **Edit:** add/remove hidden layers and neurons, choose each layer's activation, or edit individual weights and biases from the inspector. Architecture changes reset training; direct parameter edits preserve the other current weights. Inputs are fixed by the selected task (two features). Output layers support one score or two class scores. The visualization supports up to six hidden layers with twelve neurons each.
- **Examples:** XOR, a circular 2D classification boundary, and a diagonal 2D classification boundary. Selecting an example cancels only the current sample's partial updates; completed training remains intact. That selection can itself be undone.
- **Save:** store the current network, weights, random seed, and learning rate in this browser. Restore starts a fresh history at the saved weights. Saves are local; no account or backend is used. Fonts are bundled locally.

## The math

The engine in `src/engine.ts` is independent of React and uses explicit derivatives, without a machine-learning library.

For each layer:

```text
z[l] = W[l] a[l-1] + b[l]
a[l] = f(z[l])
L = 1/2 sum_j (a[last]_j - target_j)^2
```

Each sample's gradients are computed at that sample's initial weights. For elementwise activations:

```text
delta[last] = (a[last] - target) * f'(z[last])
delta[l] = (transpose(W[l+1]) delta[l+1]) * f'(z[l])
dL/dW[l] = delta[l] outer a[l-1]
dL/db[l] = delta[l]
W_new = W_old - learning_rate * dL/dW
b_new = b_old - learning_rate * dL/db
```

Supported activations: linear, tanh, sigmoid, ReLU (derivative at zero defined as zero), and numerically stable softmax. Softmax uses the full vector derivative, not an elementwise approximation:

```text
delta_j = a_j * (g_j - sum_k(g_k * a_k))
```

Softmax forward values and backward gradients depend on all neurons in their layer. Learn mode reveals those coupled results one neuron at a time.

Half squared error is used consistently for all activations. With two outputs, labels are encoded as `[1, 0]` and `[0, 1]`, and the predicted class is the larger output. With one output, classification uses a `0.5` threshold. Linear/ReLU scores are not probabilities. The loss chart averages over the full training dataset; the displayed accuracy is training accuracy, not test accuracy.

Random initialization and 2D datasets are reproducible. Computations use double precision; displayed values are rounded. Learn mode freezes the sample's learning rate for all its updates, including after undo. Undo retains the most recent 500 frames; the chart retains 400 measurements. Learn frames preserve the original sample's calculations as updates proceed. Train frames show activations for the next sample at the latest trained weights. Divergent/nonfinite weights pause training with a reset option.

## Verification

Engine tests compare every weight and bias derivative against central finite differences for all hidden/output activation combinations, verify exact parameter updates and immutable prior frames, check XOR/circle convergence, and check transitions from a partially completed lesson into training.

Browser tests exercise Learn/Train, exact weight undo/redo, save/restore, architecture edits, manual parameter edits, softmax, 2D selection, planned architecture explanations, the guide dialog, and mobile layout. They fail on console errors or failed network requests and save screenshots under `test-results/`.

## Next architecture milestones

1. CNN: small digit images, convolution kernels, feature maps, pooling, and filter gradients.
2. RNN: small sequence tasks, unrolled time steps, hidden state, and backpropagation through time.
3. LSTM: input/forget/output gates, cell state, and gate-specific gradient inspection.

The current release does not implement these architectures or image/sequence datasets. Their buttons explain this scope and do not substitute an ANN simulation.

## Documentation consulted

- [React state and effects](https://react.dev/reference/react/useEffect)
- [Vite setup](https://vite.dev/guide/)
- [Playwright web server configuration](https://playwright.dev/docs/test-webserver)
