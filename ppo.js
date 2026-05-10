class PPOAgent {
  constructor(config) {
    this.CFG = Object.assign({
      stateDim: 4,
      hiddenSize: 64,
      lr: 3e-4,
      gamma: 0.99,
      lam: 0.95,
      clipEps: 0.2,
      entropyCoef: 0.005,
      numEpochs: 10,
      batchSize: 64,
      rolloutSteps: 512,
    }, config);

    this.initNetworks();
  }

  initNetworks() {
    if (this.actor)  { this.actor.dispose(); }
    if (this.critic) { this.critic.dispose(); }
    if (this.logStd) { this.logStd.dispose(); }

    // Actor outputs μ (mean of Gaussian), 1 continuous action
    this.actor = tf.sequential({ layers: [
      tf.layers.dense({ units: this.CFG.hiddenSize, activation: 'tanh', inputShape: [this.CFG.stateDim] }),
      tf.layers.dense({ units: this.CFG.hiddenSize, activation: 'tanh' }),
      tf.layers.dense({ units: 1 }),   // μ, linear — unbounded
    ]});

    // log σ as a global learnable scalar; starts at 0 → σ = exp(0) = 1
    this.logStd = tf.variable(tf.scalar(0.0));

    this.critic = tf.sequential({ layers: [
      tf.layers.dense({ units: this.CFG.hiddenSize, activation: 'tanh', inputShape: [this.CFG.stateDim] }),
      tf.layers.dense({ units: this.CFG.hiddenSize, activation: 'tanh' }),
      tf.layers.dense({ units: 1 }),
    ]});

    this.actorOpt  = tf.train.adam(this.CFG.lr);
    this.criticOpt = tf.train.adam(this.CFG.lr);
  }

  // Box-Muller: sample from N(0,1)
  _randn() {
    const u1 = Math.random(), u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }

  // Gaussian log probability: log N(a; μ, σ)
  _gaussianLogProb(a, mu, sigma) {
    const LOG2PI = Math.log(2 * Math.PI);
    return -0.5 * ((a - mu) / sigma) ** 2 - Math.log(sigma) - 0.5 * LOG2PI;
  }

  selectAction(state) {
    // Get μ from actor network
    const muVal = tf.tidy(() => {
      const out = this.actor.predict(tf.tensor2d([state])).squeeze();
      return out.arraySync();
    });
    const mu = Array.isArray(muVal) ? muVal[0] : muVal;

    // Clamp log σ to avoid numerical instability
    const logStdVal = Math.max(-2, Math.min(2, this.logStd.arraySync()));
    const sigma = Math.exp(logStdVal);

    // Sample action: a = μ + σ·ε,  ε ~ N(0,1)
    const eps    = this._randn();
    const action = mu + sigma * eps;

    const logProb = this._gaussianLogProb(action, mu, sigma);

    const value = tf.tidy(() => this.critic.predict(tf.tensor2d([state])).squeeze().arraySync());
    return { action, logProb, value: typeof value === 'number' ? value : value[0] };
  }

  computeGAE(rewards, values, dones, lastValue) {
    const n   = rewards.length;
    const adv = new Array(n);
    let gaeNext = 0;

    for (let t = n - 1; t >= 0; t--) {
      const mask    = dones[t] ? 0 : 1;
      const nextVal = dones[t] ? 0 : (t === n - 1 ? lastValue : values[t + 1]);
      const delta   = rewards[t] + this.CFG.gamma * nextVal - values[t];
      gaeNext       = delta + this.CFG.gamma * this.CFG.lam * mask * gaeNext;
      adv[t]        = gaeNext;
    }
    const ret = adv.map((a, i) => a + values[i]);
    return { advantages: adv, returns: ret };
  }

  async update(states, actions, oldLogProbs, advantages, returns) {
    // Normalize advantages
    const mean = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const vari = advantages.reduce((a, b) => a + (b - mean) ** 2, 0) / advantages.length;
    const std  = Math.sqrt(vari + 1e-8);
    const normAdv = advantages.map(a => (a - mean) / std);

    const n = states.length;
    let sumActorLoss = 0, sumCriticLoss = 0, cnt = 0;

    for (let epoch = 0; epoch < this.CFG.numEpochs; epoch++) {
      // Fisher-Yates shuffle
      const idx = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }

      for (let start = 0; start < n; start += this.CFG.batchSize) {
        const bi = idx.slice(start, Math.min(start + this.CFG.batchSize, n));

        // Build batch tensors (actions are now floats)
        const bStates  = tf.tensor2d(bi.map(i => states[i]));
        const bActions = tf.tensor2d(bi.map(i => [actions[i]]));  // [B, 1] float
        const bOldLP   = tf.tensor1d(bi.map(i => oldLogProbs[i]));
        const bAdv     = tf.tensor1d(bi.map(i => normAdv[i]));
        const bRet     = tf.tensor1d(bi.map(i => returns[i]));

        // ── Actor loss (PPO-Clip + Gaussian Entropy) ──
        const aLoss = this.actorOpt.minimize(() => {
          const mu      = this.actor.predict(bStates);                   // [B, 1]
          const logStd  = tf.clipByValue(this.logStd, -2, 2);           // scalar
          const sigma   = tf.exp(logStd);

          // log π(a|s) = -0.5·((a-μ)/σ)² - log(σ) - 0.5·log(2π)
          const diff        = tf.sub(bActions, mu);                      // [B, 1]
          const actLogProbs = tf.squeeze(tf.sub(
            tf.mul(-0.5, tf.square(tf.div(diff, sigma))),
            tf.add(logStd, tf.scalar(0.5 * Math.log(2 * Math.PI)))
          ));                                                             // [B]

          const ratio      = tf.exp(tf.sub(actLogProbs, bOldLP));
          const surr1      = tf.mul(ratio, bAdv);
          const surr2      = tf.mul(tf.clipByValue(ratio, 1 - this.CFG.clipEps, 1 + this.CFG.clipEps), bAdv);
          const policyLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));

          // Gaussian entropy: H = 0.5 + 0.5·log(2π) + log(σ)
          const entropy = tf.add(
            tf.scalar(0.5 * (1 + Math.log(2 * Math.PI))),
            logStd
          );

          return tf.sub(policyLoss, tf.mul(tf.scalar(this.CFG.entropyCoef), entropy));
        }, true, [...this.actor.trainableWeights.map(w => w.val), this.logStd]);

        // ── Critic loss (MSE) ──
        const cLoss = this.criticOpt.minimize(() => {
          const vals = this.critic.predict(bStates).squeeze();
          return tf.mean(tf.square(tf.sub(bRet, vals)));
        }, true);

        if (aLoss && cLoss) {
          sumActorLoss  += aLoss.arraySync();
          sumCriticLoss += cLoss.arraySync();
          cnt++;
          tf.dispose([aLoss, cLoss]);
        }

        tf.dispose([bStates, bActions, bOldLP, bAdv, bRet]);
      }

      await new Promise(r => setTimeout(r, 0));
    }

    return {
      actorLoss:  cnt > 0 ? sumActorLoss  / cnt : 0,
      criticLoss: cnt > 0 ? sumCriticLoss / cnt : 0,
    };
  }

  getValue(state) {
    return tf.tidy(() => this.critic.predict(tf.tensor2d([state])).squeeze().arraySync());
  }

  // Batch inference: one GPU forward pass for all N envs at once
  selectActionBatch(states) {
    const muArr = tf.tidy(() => {
      const out = this.actor.predict(tf.tensor2d(states)); // [N, 1]
      return out.arraySync();
    });
    const valArr = tf.tidy(() => {
      const out = this.critic.predict(tf.tensor2d(states)).squeeze();
      const arr = out.arraySync();
      return Array.isArray(arr) ? arr : [arr];
    });

    const logStdVal = Math.max(-2, Math.min(2, this.logStd.arraySync()));
    const sigma = Math.exp(logStdVal);

    const actions = [], logProbs = [], values = [];
    for (let i = 0; i < states.length; i++) {
      const mu  = Array.isArray(muArr[i]) ? muArr[i][0] : muArr[i];
      const eps = this._randn();
      const a   = mu + sigma * eps;
      actions.push(a);
      logProbs.push(this._gaussianLogProb(a, mu, sigma));
      values.push(valArr[i]);
    }
    return { actions, logProbs, values };
  }

  // Batch value lookup for last-value bootstrap
  getValueBatch(states) {
    return tf.tidy(() => {
      const out = this.critic.predict(tf.tensor2d(states)).squeeze();
      const arr = out.arraySync();
      return Array.isArray(arr) ? arr : [arr];
    });
  }
}

window.PPOAgent = PPOAgent;
