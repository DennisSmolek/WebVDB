/**
 * A small native control panel for demo 10 (no GUI dependency): labelled `<select>`, range and
 * checkbox rows that call back on change. Every input gets the id `ctl-<key>`, which the e2e spec
 * drives. `set()` updates a row without firing its callback; `show()` / `enable()` gate rows by scene.
 */

export interface PanelRow<T> {
  readonly el: HTMLDivElement;
  readonly input: HTMLInputElement | HTMLSelectElement;
  set(value: T): void;
  show(visible: boolean): void;
  enable(enabled: boolean): void;
}

export interface SelectOption<T extends string> {
  value: T;
  label: string;
}

export class Panel {
  constructor(private readonly root: HTMLElement) {}

  heading(text: string): HTMLHeadingElement {
    const h = document.createElement("h2");
    h.textContent = text;
    this.root.append(h);
    return h;
  }

  select<T extends string>(key: string, label: string, options: readonly SelectOption<T>[], value: T, onChange: (v: T) => void): PanelRow<T> & { options(next: readonly SelectOption<T>[]): void } {
    const select = document.createElement("select");
    const fill = (opts: readonly SelectOption<T>[]): void => {
      select.replaceChildren(
        ...opts.map((o) => {
          const el = document.createElement("option");
          el.value = o.value;
          el.textContent = o.label;
          return el;
        }),
      );
    };
    fill(options);
    select.value = value;
    select.addEventListener("change", () => onChange(select.value as T));
    const row = this.row(key, label, select, null, true);
    return {
      ...row,
      set: (v: T) => {
        select.value = v;
      },
      options: (next) => {
        const current = select.value;
        fill(next);
        select.value = next.some((o) => o.value === current) ? current : (next[0]?.value ?? "");
      },
    };
  }

  range(key: string, label: string, min: number, max: number, step: number, value: number, onChange: (v: number) => void, format?: (v: number) => string): PanelRow<number> {
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    const out = document.createElement("output");
    const fmt = format ?? ((v: number) => v.toFixed(step < 1 ? 2 : 0));
    out.textContent = fmt(value);
    input.addEventListener("input", () => {
      out.textContent = fmt(Number(input.value));
      onChange(Number(input.value));
    });
    const row = this.row(key, label, input, out, false);
    return {
      ...row,
      set: (v: number) => {
        input.value = String(v);
        out.textContent = fmt(v);
      },
    };
  }

  checkbox(key: string, label: string, value: boolean, onChange: (v: boolean) => void): PanelRow<boolean> {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = value;
    input.style.justifySelf = "start";
    input.addEventListener("change", () => onChange(input.checked));
    const row = this.row(key, label, input, null, true);
    return {
      ...row,
      set: (v: boolean) => {
        input.checked = v;
      },
    };
  }

  private row(key: string, label: string, input: HTMLInputElement | HTMLSelectElement, out: HTMLOutputElement | null, wide: boolean): Omit<PanelRow<never>, "set"> {
    const el = document.createElement("div");
    el.className = wide ? "ctl wide" : "ctl";
    input.id = `ctl-${key}`;
    const lab = document.createElement("label");
    lab.htmlFor = input.id;
    lab.textContent = label;
    el.append(lab, input);
    if (out) el.append(out);
    this.root.append(el);
    return {
      el,
      input,
      show: (visible: boolean) => {
        el.hidden = !visible;
      },
      enable: (enabled: boolean) => {
        input.disabled = !enabled;
        el.classList.toggle("disabled", !enabled);
      },
    };
  }
}
