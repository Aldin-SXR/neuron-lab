# Neuron Lab

A browser-based neural network playground for university beginners. Four architectures are implemented end to end, each with exact, explicit gradients and a step-by-step Learn mode: feedforward ANN, CNN, RNN, and LSTM. The interface is available in English and Bosnian.

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

If Chromium is not installed, run `npx playwright install chromium` before browser tests. To use an existing Chromium binary instead, set `CHROMIUM_PATH=/path/to/chromium`.

## Explore

- **First visit:** a welcome dialog offers a language choice and a one-minute guided tour that spotlights each part of the interface. The tour and the guide can be reopened from the header at any time.
- **Language and text size:** switch between English (EN) and Bosnian (BS) in the header. The choice is remembered, and the page's `lang` attribute follows it. Bosnian prose and metrics use a decimal comma; formulas keep the decimal point. The A−/A+ control scales all text, including the diagrams (100%, 112.5%, 125%).
- **Learn:** step through every calculation of one example: forward pass, loss, backpropagation, then the parameter updates. “Next step”, “Back” (exact undo), and “Play” are labeled buttons; ← / → / space work as keyboard shortcuts. Inside the network picker and tabs, arrow keys move between options instead. A hint under each step says what to do next.
- **Train:** train one epoch per step/tick using stochastic gradient descent. Switching from a partially completed lesson finishes its pending updates before training the epoch, so the first epoch count may be fractional.
- **Inspect:** click any neuron, connection, feature-map cell, filter, or time step to see its formula with the actual numbers.
- **Save:** each architecture saves its own network, weights, settings, and learning rate in this browser. Restore starts a fresh history at the saved weights. No account or backend is used, and fonts are bundled locally.

### ANN (feedforward)

- XOR, a circular 2D classification boundary, and a diagonal 2D boundary. Selecting a problem also selects its recommended learning rate (0.3 for XOR, 0.1 for the 2D problems).
- Add/remove hidden layers (up to 6) and neurons (up to 12), choose each layer's activation, or edit a neuron's weights and bias directly. Learn mode updates one weight or bias per step.

### CNN (images)

- 6×6 grayscale images: *lines* (horizontal, vertical, diagonal; 24 images) or *shapes* (plus, X, square in all 16 positions; 48 images), with reproducible noise.
- 1–4 filters of 3×3 (stride 1, no padding) → ReLU or tanh → 2×2 max or average pooling → dense layer → softmax over 3 classes.
- Clicking a feature-map cell highlights its 3×3 receptive field in the input. During backpropagation the view switches to gradients, and max-pooling winners are marked. Learn mode updates one filter or one class's dense weights per step.
- A gallery shows each training image with its current prediction. “Draw your own” lets learners paint a 6×6 image and see the network's probabilities.

### RNN and LSTM (sequences)

- *Predict the next letter* of any word the learner types (3–12 letters, 2–8 distinct letters; default “hello”). Repeated letters with different successors require memory. “Let it write” feeds the network's own greedy predictions back in.
- *Remember the first bit* in a 0/1 sequence of length 2–8. All sequences are enumerated up to length 5; longer lengths use 32 reproducible sequences balanced by first bit. The answer is needed only at the last step.
- The network is shown unrolled in time: prediction probabilities, LSTM gates (f, i, o) and cell state, hidden state, input, and, during backpropagation through time, the size of ‖∂L/∂h‖ at each step.
- 1–8 hidden units. Learn mode updates one parameter group per step: the RNN's input weights, recurrent weights, and output weights; the LSTM's forget, input, candidate, and output gates, then its output weights.

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

### CNN, RNN, and LSTM

These engines (`src/cnn.ts`, `src/sequence.ts`) share the lesson and training machinery in `src/lab.ts`. They use softmax with cross-entropy, so the gradient of each output score is `p − y`.

```text
CNN:   z[k](r,c) = Σ_uv W[k](u,v) x(r+u, c+v) + b[k];  a = f(z);  p = pool2x2(a)
       max pooling routes each gradient to the winning cell; average pooling shares it equally
       dL/dW[k](u,v) = Σ_rc dL/dz[k](r,c) · x(r+u, c+v)
RNN:   h_t = tanh(Wx x_t + Wh h_{t-1} + b);  y_t = softmax(Wy h_t + by)
LSTM:  f, i, o = σ(W· [x_t, h_{t-1}] + b);  g = tanh(Wg [x_t, h_{t-1}] + bg)
       c_t = f ⊙ c_{t-1} + i ⊙ g;  h_t = o ⊙ tanh(c_t)
Loss (sequences) = mean over time steps that have a target of −log p_t(target)
```

Backpropagation through time sums each weight's gradient over all time steps, with no gradient clipping. The LSTM forget-gate bias starts at 1. Divergent/nonfinite parameters pause training with a reset option.

## Verification

`npm test` runs the engine tests:

- **Gradient checks:** every analytical derivative is compared against a central finite difference. This covers all ANN activation combinations, CNN with both activations and both pooling types, and RNN/LSTM on both sequence tasks.
- **Lesson updates:** each lesson step must change exactly its highlighted parameters, by the stored gradients, without mutating earlier frames.
- **Convergence:** XOR, the circle, both CNN problems, and both sequence tasks must learn.
- **Partial lessons:** switching from a partial lesson to training must first finish its pending updates.
- **Translation:** the Bosnian dictionary must have the same structure as the English one, with no empty text and correct plural forms.

`npm run test:browser` runs Chromium user flows. They cover:

- Learn/Train, exact undo/redo, and save/restore.
- ANN architecture and parameter edits, and softmax.
- A CNN lesson through the filter updates, then training and the drawing pad.
- RNN learning and writing back a custom word.
- LSTM in Bosnian, including language persistence.
- The welcome dialog, guided tour, guide, and text size.
- Mobile layout for all four networks.

The tests fail on console errors or failed network requests and save screenshots under `test-results/`.

## Documentation consulted

- [React state and effects](https://react.dev/reference/react/useEffect)
- [Vite setup](https://vite.dev/guide/)
- [Playwright web server configuration](https://playwright.dev/docs/test-webserver)
