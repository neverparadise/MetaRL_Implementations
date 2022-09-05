a = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500]
print(len(a))


env_names = ['assembly-v2-goal-hidden', 'basketball-v2-goal-hidden', 'bin-picking-v2-goal-hidden', 'box-close-v2-goal-hidden',
             'button-press-topdown-v2-goal-hidden', 'button-press-topdown-wall-v2-goal-hidden',
             'button-press-v2-goal-hidden', 'button-press-wall-v2-goal-hidden', 'coffee-button-v2-goal-hidden',
             'coffee-pull-v2-goal-hidden', 'coffee-push-v2-goal-hidden', 'dial-turn-v2-goal-hidden',
             'disassemble-v2-goal-hidden', 'door-close-v2-goal-hidden', 'door-lock-v2-goal-hidden', 'door-open-v2-goal-hidden',
             'door-unlock-v2-goal-hidden', 'hand-insert-v2-goal-hidden', 'drawer-close-v2-goal-hidden',
             'drawer-open-v2-goal-hidden', 'faucet-open-v2-goal-hidden', 'faucet-close-v2-goal-hidden', 'hammer-v2-goal-hidden',
             'handle-press-side-v2-goal-hidden', 'handle-press-v2-goal-hidden', 'handle-pull-side-v2-goal-hidden',
             'handle-pull-v2-goal-hidden', 'lever-pull-v2-goal-hidden', 'peg-insert-side-v2-goal-hidden',
             'pick-place-wall-v2-goal-hidden', 'pick-out-of-hole-v2-goal-hidden', 'reach-v2-goal-hidden',
             'push-back-v2-goal-hidden', 'push-v2-goal-hidden', 'pick-place-v2-goal-hidden', 'plate-slide-v2-goal-hidden',
             'plate-slide-side-v2-goal-hidden', 'plate-slide-back-v2-goal-hidden',
             'plate-slide-back-side-v2-goal-hidden', 'peg-unplug-side-v2-goal-hidden', 'soccer-v2-goal-hidden',
             'stick-push-v2-goal-hidden', 'stick-pull-v2-goal-hidden', 'push-wall-v2-goal-hidden', 'reach-wall-v2-goal-hidden',
             'shelf-place-v2-goal-hidden', 'sweep-into-v2-goal-hidden', 'sweep-v2-goal-hidden', 'window-open-v2-goal-hidden',
             'window-close-v2-goal-hidden']

env_names = ['assembly-v2-goal-observable', 'basketball-v2-goal-observable', 'bin-picking-v2-goal-observable', 'box-close-v2-goal-observable',
             'button-press-topdown-v2-goal-observable', 'button-press-topdown-wall-v2-goal-observable',
             'button-press-v2-goal-observable', 'button-press-wall-v2-goal-observable', 'coffee-button-v2-goal-observable',
             'coffee-pull-v2-goal-observable', 'coffee-push-v2-goal-observable', 'dial-turn-v2-goal-observable',
             'disassemble-v2-goal-observable', 'door-close-v2-goal-observable', 'door-lock-v2-goal-observable', 'door-open-v2-goal-observable',
             'door-unlock-v2-goal-observable', 'hand-insert-v2-goal-observable', 'drawer-close-v2-goal-observable',
             'drawer-open-v2-goal-observable', 'faucet-open-v2-goal-observable', 'faucet-close-v2-goal-observable', 'hammer-v2-goal-observable',
             'handle-press-side-v2-goal-observable', 'handle-press-v2-goal-observable', 'handle-pull-side-v2-goal-observable',
             'handle-pull-v2-goal-observable', 'lever-pull-v2-goal-observable', 'peg-insert-side-v2-goal-observable',
             'pick-place-wall-v2-goal-observable', 'pick-out-of-hole-v2-goal-observable', 'reach-v2-goal-observable',
             'push-back-v2-goal-observable', 'push-v2-goal-observable', 'pick-place-v2-goal-observable', 'plate-slide-v2-goal-observable',
             'plate-slide-side-v2-goal-observable', 'plate-slide-back-v2-goal-observable',
             'plate-slide-back-side-v2-goal-observable', 'peg-unplug-side-v2-goal-observable', 'soccer-v2-goal-observable',
             'stick-push-v2-goal-observable', 'stick-pull-v2-goal-observable', 'push-wall-v2-goal-observable', 'reach-wall-v2-goal-observable',
             'shelf-place-v2-goal-observable', 'sweep-into-v2-goal-observable', 'sweep-v2-goal-observable', 'window-open-v2-goal-observable',
             'window-close-v2-goal-observable']

print(len(env_names))