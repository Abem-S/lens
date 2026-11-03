/**
 * One page check at a time, without losing a step: a check asked for while another
 * is running waits (the newest replaces any older waiting one) and goes as soon as
 * the running check finishes. Dropping it instead meant a step finished during a
 * slow read was never checked.
 */
export class CheckQueue<T> {
  private running = false;
  private waiting: T | null = null;

  get busy(): boolean {
    return this.running;
  }

  /** Returns the item to send now, or null when it waits for the running check. */
  offer(item: T): T | null {
    if (this.running) {
      this.waiting = item;
      return null;
    }
    this.running = true;
    return item;
  }

  /** The running check finished: returns the waiting item to send now, if any. */
  done(): T | null {
    const next = this.waiting;
    this.waiting = null;
    this.running = next !== null;
    return next;
  }
}
