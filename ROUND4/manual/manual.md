# **Manual trading challenge: “Vanilla Just Isn’t Exotic Enough”**

As the Intarian economy evolved, trading expanded beyond standard calls and puts. In this round, you can trade `AETHER_CRYSTAL`, vanilla options with 2 and 3 week expiries, and several exotic derivatives written on the same underlying. Please note that a ‘week’ here refers to 5 trading days and that the ‘standard’ number of trading days per year is 252 (since some big exchanges are typically open 252 days per year). 

Your objective is to construct positions that generate positive expected PnL. But be aware: unhedged exposure can lead to large losses, so risk management matters.

All products are written on `AETHER_CRYSTAL`. You can trade the underlying, 2 week and 3 week vanilla calls and puts, and the following exotics:

<aside>
❓

### **Chooser Option**

Expires in 3 weeks. After 2 weeks, the buyer chooses whether it becomes a call or a put, selecting whichever would be in the money at that time. It then behaves like a standard option for the final week until expiry.

</aside>

<aside>
🔀

### Binary Put Option

Has an all-or-nothing payoff. If the underlying is below the strike at expiry, it pays the specified amount. Otherwise, it expires worthless.

</aside>

<aside>
🥊

### **Knock-Out Put Option**

Behaves like a regular put unless the underlying ever trades below the knockout barrier before expiry. If the barrier is breached at any point, the option immediately becomes worthless.

</aside>

You may buy or sell up to the displayed volume in each product. Your final score is the average PnL across 100 simulations of the underlying.

The underlying `AETHER_CRYSTAL` is simulated using Geometric Brownian Motion with zero risk-neutral drift and fixed annualized volatility of 251%. Prices evolve on a discrete grid of 4 steps per trading day, assuming 252 trading days per year.

And remember, when payoffs become conditional, so does risk. Good luck!



AC_50_CO is an Aether Crystal CHOOSER Option contract with a Strike Price of 50 XIRECs and a Time To Expiry of 21 Solvenarian Days (starting from Round 1, on Intara). After 14 Solvenarian Days, the buyer chooses the side (PUT or CALL). At that point, the contract automatically converts to the side that is “in the money”. After the remaining 7 Solvenarian Days, the contract expires like a standard PUT or CALL option.

AC_40_BP is an Aether Crystal BINARY PUT Option contract with a Strike Price of 40 XIRECs and a Time To Expiry of 21 Solvenarian Days (starting from Round 1, on Intara). If the value of the Aether Crystal at expiry is below 40 XIRECs, the contract pays a fixed amount of 10 XIRECs. If the value is at or above 40 XIRECs at expiry, the contract expires worthless.

AC_45_KO is an Aether Crystal KNOCK-OUT PUT Option contract with a Strike Price of 45 XIRECs, a Barrier Price of 35 XIRECs, and a Time To Expiry of 21 Solvenarian Days (starting from Round 1, on Intara). If the value of the Aether Crystal ever falls below 35 XIRECs, the contract is knocked out and expires worthless. If the barrier is never breached, the contract expires with the same payoff as a standard put option with a Strike Price of 45 XIRECs.