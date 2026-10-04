import { useLayoutEffect, useRef } from 'react'

const ON = ':scope > [aria-selected="true"], :scope > [aria-pressed="true"], :scope > [aria-checked="true"], :scope > .is-on'

/*
  Sliding highlight for a row of tabs or segment buttons. Drop it in as the first child of the container.
  It follows whichever sibling is selected. The edge in the direction of travel leaves first and the
  trailing edge catches up a beat later, so the highlight stretches and settles like a blob.
*/
export function SlideInd() {
  const ref = useRef<HTMLSpanElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    const box = el?.parentElement
    if (!el || !box) return
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let prev: number | null = null
    const place = () => {
      const on = box.querySelector<HTMLElement>(ON)
      if (!on) {
        el.style.opacity = '0'
        return
      }
      const l = on.offsetLeft
      const r = box.clientWidth - (on.offsetLeft + on.offsetWidth)
      if (prev == null || reduce) {
        el.style.transition = 'none'
      } else if (l !== prev) {
        const lead = 'cubic-bezier(.3,.9,.25,1)'
        const trail = 'cubic-bezier(.6,0,.2,1)'
        el.style.transition =
          l > prev
            ? `right .28s ${lead}, left .44s ${trail} .05s, top .3s, height .3s, opacity .2s`
            : `left .28s ${lead}, right .44s ${trail} .05s, top .3s, height .3s, opacity .2s`
      }
      el.style.left = `${l}px`
      el.style.right = `${r}px`
      el.style.top = `${on.offsetTop}px`
      el.style.height = `${on.offsetHeight}px`
      el.style.opacity = on.matches(':disabled') && on.closest('.seg') ? '0.55' : '1'
      prev = l
    }
    place()
    const mo = new MutationObserver(place)
    mo.observe(box, { subtree: true, attributes: true, attributeFilter: ['aria-selected', 'aria-pressed', 'aria-checked', 'class', 'disabled'] })
    const ro = new ResizeObserver(place)
    ro.observe(box)
    return () => {
      mo.disconnect()
      ro.disconnect()
    }
  }, [])
  return <span ref={ref} className="slide-ind" aria-hidden="true" />
}
