import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js'

/*
  Ambient 3D sorting line for the sign-in screen: parcels ride a belt through a scanner gate,
  doubtful ones are pushed down a chute into the hold bay. Purely decorative and interactive
  (orbit, zoom, hover); it scores nothing and talks to no API.
*/

type Act = 'allow' | 'scan' | 'owner' | 'review' | 'hold' | 'block'
const HEX: Record<Act, number> = { allow: 0x2bd48a, scan: 0x2fd3ea, owner: 0x6f93ff, review: 0xa67cff, hold: 0xff6a2b, block: 0xff3b5c }
const NAME: Record<Act, string> = { allow: 'Allowed', scan: 'Scan check', owner: 'Ask owner', review: 'Review', hold: 'Hold', block: 'Block' }
const STATES = ['SP', 'RJ', 'MG', 'BA', 'PR', 'RS', 'PE', 'CE', 'GO', 'SC']
const rnd = <T,>(a: T[]) => a[Math.floor(Math.random() * a.length)]
const hexs = (n: number) => '#' + n.toString(16).padStart(6, '0')

type PState = 'belt' | 'chute' | 'bay'
interface ParcelData {
  w: number
  h: number
  d: number
  box: THREE.Mesh
  tag: THREE.Mesh
  state: PState
  acc: string
  dest: string
  act: Act | null
  t: number
  from: THREE.Vector3
  bayIdx: number
  phase: number
}

export interface TerminalScene {
  dispose: () => void
}

export function createTerminalScene(host: HTMLElement, canvas: HTMLCanvasElement, tip: HTMLElement): TerminalScene {
  const T = THREE
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const R = new T.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' })
  R.setPixelRatio(Math.min(2, window.devicePixelRatio || 1))
  R.shadowMap.enabled = true
  R.shadowMap.type = T.PCFSoftShadowMap
  R.toneMapping = T.ACESFilmicToneMapping
  R.toneMappingExposure = 0.95

  const scene = new T.Scene()
  scene.background = new T.Color(0x07090c)
  scene.fog = new T.FogExp2(0x07090c, 0.032)
  const cam = new T.PerspectiveCamera(36, 1, 0.1, 200)
  cam.position.set(11.5, 7.2, 12.5)

  const ctl = new OrbitControls(cam, canvas)
  ctl.target.set(0.8, 0.9, 0.8)
  ctl.enableDamping = true
  ctl.dampingFactor = 0.07
  ctl.enablePan = false
  ctl.minDistance = 7
  ctl.maxDistance = 26
  ctl.maxPolarAngle = 1.38
  ctl.minPolarAngle = 0.35
  ctl.autoRotate = !reduce
  ctl.autoRotateSpeed = 0.35
  ctl.update()

  const disposables: { dispose: () => void }[] = []
  const keep = <D extends { dispose: () => void }>(d: D) => (disposables.push(d), d)

  // Light: three work lamps over the line, a cool rim from behind.
  scene.add(new T.HemisphereLight(0x8fa3c4, 0x0a0c10, 1.3))
  const spot = (x: number, z: number, i: number) => {
    const s = new T.SpotLight(0xfff1df, i, 30, 0.55, 0.55, 0)
    s.position.set(x, 9, z)
    s.target.position.set(x, 0, z)
    s.castShadow = true
    s.shadow.mapSize.set(1024, 1024)
    s.shadow.bias = -0.0004
    scene.add(s, s.target)
    const cone = new T.Mesh(
      keep(new T.ConeGeometry(2.6, 9, 40, 1, true)),
      keep(new T.MeshBasicMaterial({ color: 0xffe7c9, transparent: true, opacity: 0.018, depthWrite: false, side: T.DoubleSide, blending: T.AdditiveBlending })),
    )
    cone.position.set(x, 4.5, z)
    scene.add(cone)
    const lamp = new T.Mesh(keep(new T.CylinderGeometry(0.45, 0.6, 0.25, 24)), keep(new T.MeshStandardMaterial({ color: 0x22262c })))
    lamp.position.set(x, 9, z)
    scene.add(lamp)
    const bulb = new T.Mesh(keep(new T.CircleGeometry(0.42, 24)), keep(new T.MeshBasicMaterial({ color: 0xfff3e2 })))
    bulb.rotation.x = Math.PI / 2
    bulb.position.set(x, 8.87, z)
    scene.add(bulb)
  }
  spot(-6, 0, 2.4)
  spot(0, 0.4, 3.0)
  spot(6, 1.4, 2.4)
  const rim = new T.DirectionalLight(0x6f93ff, 1.1)
  rim.position.set(-8, 5, -9)
  scene.add(rim)

  // Depot floor: speckled concrete with a faint bay grid and painted lines.
  const canvasTex = (w: number, h: number, draw: (x: CanvasRenderingContext2D) => void) => {
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    draw(c.getContext('2d')!)
    const t = keep(new T.CanvasTexture(c))
    t.colorSpace = T.SRGBColorSpace
    return t
  }
  const ft = canvasTex(1024, 1024, (x) => {
    x.fillStyle = '#15181d'
    x.fillRect(0, 0, 1024, 1024)
    for (let i = 0; i < 14000; i++) {
      const v = Math.random() * 30
      x.fillStyle = `rgba(${v + 20},${v + 24},${v + 30},.25)`
      x.fillRect(Math.random() * 1024, Math.random() * 1024, 2, 2)
    }
    x.strokeStyle = 'rgba(255,255,255,.05)'
    x.lineWidth = 2
    for (let i = 0; i <= 1024; i += 128) {
      x.beginPath()
      x.moveTo(i, 0)
      x.lineTo(i, 1024)
      x.stroke()
      x.beginPath()
      x.moveTo(0, i)
      x.lineTo(1024, i)
      x.stroke()
    }
  })
  ft.wrapS = ft.wrapT = T.RepeatWrapping
  ft.repeat.set(8, 8)
  const floor = new T.Mesh(keep(new T.PlaneGeometry(90, 90)), keep(new T.MeshStandardMaterial({ map: ft, roughness: 0.62, metalness: 0.25 })))
  floor.rotation.x = -Math.PI / 2
  floor.receiveShadow = true
  scene.add(floor)
  const paint = keep(new T.MeshBasicMaterial({ color: 0x9aa3b0, transparent: true, opacity: 0.28 }))
  ;[-1.75, 1.75].forEach((z) => {
    const m = new T.Mesh(keep(new T.PlaneGeometry(30, 0.06)), paint)
    m.rotation.x = -Math.PI / 2
    m.position.set(0, 0.003, z)
    scene.add(m)
  })
  for (let i = -6; i < 7; i++) {
    const m = new T.Mesh(keep(new T.PlaneGeometry(0.5, 0.06)), paint)
    m.rotation.x = -Math.PI / 2
    m.rotation.z = 0.6
    m.position.set(i * 2.2, 0.003, -2.4)
    scene.add(m)
  }

  // Conveyor
  const BELT_Y = 0.82
  const BELT_W = 1.5
  const X0 = -16
  const X1 = 16
  const steel = keep(new T.MeshStandardMaterial({ color: 0x2a2f37, metalness: 0.85, roughness: 0.35 }))
  const steelL = keep(new T.MeshStandardMaterial({ color: 0x4a515c, metalness: 0.9, roughness: 0.28 }))
  const bt = canvasTex(512, 64, (x) => {
    x.fillStyle = '#101215'
    x.fillRect(0, 0, 512, 64)
    for (let i = 0; i < 512; i += 16) {
      x.fillStyle = '#1d2127'
      x.fillRect(i, 0, 6, 64)
    }
  })
  bt.wrapS = bt.wrapT = T.RepeatWrapping
  bt.repeat.set(8, 1)
  const belt = new T.Mesh(keep(new T.BoxGeometry(X1 - X0, 0.08, BELT_W)), keep(new T.MeshStandardMaterial({ map: bt, roughness: 0.85, metalness: 0.1 })))
  belt.position.set(0, BELT_Y - 0.04, 0)
  belt.receiveShadow = true
  scene.add(belt)
  ;[-1, 1].forEach((sd) => {
    const rail = new T.Mesh(keep(new T.BoxGeometry(X1 - X0, 0.22, 0.08)), steelL)
    rail.position.set(0, BELT_Y + 0.04, sd * (BELT_W / 2 + 0.05))
    rail.castShadow = true
    scene.add(rail)
    const skirt = new T.Mesh(keep(new T.BoxGeometry(X1 - X0, 0.32, 0.04)), steel)
    skirt.position.set(0, BELT_Y - 0.24, sd * (BELT_W / 2 + 0.02))
    scene.add(skirt)
  })
  const dm = new T.Object3D()
  const legs = new T.InstancedMesh(keep(new T.BoxGeometry(0.09, BELT_Y - 0.1, 0.09)), steel, 34)
  let li = 0
  for (let x = X0 + 1; x < X1; x += 2) {
    for (const sd of [-1, 1]) {
      dm.position.set(x, (BELT_Y - 0.1) / 2, sd * (BELT_W / 2 - 0.05))
      dm.updateMatrix()
      legs.setMatrixAt(li++, dm.matrix)
    }
  }
  legs.count = li
  legs.castShadow = true
  scene.add(legs)
  const rolG = keep(new T.CylinderGeometry(0.06, 0.06, BELT_W + 0.12, 12))
  rolG.rotateX(Math.PI / 2)
  const rollers = new T.InstancedMesh(rolG, steelL, 64)
  for (let i = 0; i < 64; i++) {
    dm.position.set(X0 + 0.25 + i * 0.5, BELT_Y - 0.13, 0)
    dm.updateMatrix()
    rollers.setMatrixAt(i, dm.matrix)
  }
  scene.add(rollers)

  // Diverter arm, chute and the hold bay
  const DIV_X = 3.2
  const BAY = new T.Vector3(6.4, 0.02, 4.6)
  const chute = new T.Mesh(keep(new T.BoxGeometry(4.6, 0.06, 1.2)), keep(new T.MeshStandardMaterial({ color: 0x1a1e24, metalness: 0.6, roughness: 0.4 })))
  chute.position.set(4.9, 0.5, 2.3)
  chute.rotation.y = -0.62
  chute.rotation.z = -0.12
  chute.receiveShadow = true
  scene.add(chute)
  const armPivot = new T.Group()
  armPivot.position.set(DIV_X - 0.95, 0, -0.7)
  scene.add(armPivot)
  const arm = new T.Mesh(keep(new T.BoxGeometry(1.9, 0.24, 0.06)), keep(new T.MeshStandardMaterial({ color: 0x8a94a3, metalness: 0.9, roughness: 0.25 })))
  arm.position.set(0.95, BELT_Y + 0.17, 0)
  armPivot.add(arm)
  const bayRingMat = keep(new T.MeshBasicMaterial({ color: HEX.hold, transparent: true, opacity: 0.55, side: T.DoubleSide, blending: T.AdditiveBlending, depthWrite: false }))
  const bayRing = new T.Mesh(keep(new T.RingGeometry(1.7, 1.82, 72)), bayRingMat)
  bayRing.rotation.x = -Math.PI / 2
  bayRing.position.set(BAY.x, 0.01, BAY.z)
  scene.add(bayRing)
  const bayFill = new T.Mesh(keep(new T.CircleGeometry(1.7, 72)), keep(new T.MeshBasicMaterial({ color: HEX.hold, transparent: true, opacity: 0.06, depthWrite: false })))
  bayFill.rotation.x = -Math.PI / 2
  bayFill.position.set(BAY.x, 0.008, BAY.z)
  scene.add(bayFill)

  const sprite = (text: string, col: number) => {
    const t = canvasTex(1024, 128, (x) => {
      x.font = '600 52px "IBM Plex Mono", monospace'
      const w = Math.min(1010, x.measureText(text).width + 56)
      x.fillStyle = 'rgba(7,9,12,.82)'
      x.fillRect((1024 - w) / 2, 18, w, 92)
      x.strokeStyle = hexs(col)
      x.lineWidth = 4
      x.strokeRect((1024 - w) / 2 + 2, 20, w - 4, 88)
      x.fillStyle = hexs(col)
      x.textAlign = 'center'
      x.fillText(text, 512, 82)
    })
    const sp = new T.Sprite(keep(new T.SpriteMaterial({ map: t, depthTest: false, transparent: true })))
    sp.scale.set(2.4, 0.3, 1)
    sp.renderOrder = 10
    return sp
  }
  const bayLabel = sprite('HOLD BAY · NO LABEL', HEX.hold)
  bayLabel.position.set(BAY.x, 2.7, BAY.z)
  bayLabel.scale.set(3.2, 0.4, 1)
  scene.add(bayLabel)

  // Scanner gate with a light curtain shader
  const gate = new T.Group()
  scene.add(gate)
  const frameM = keep(new T.MeshStandardMaterial({ color: 0x1b1f25, metalness: 0.7, roughness: 0.35 }))
  ;[-1, 1].forEach((sd) => {
    const p = new T.Mesh(keep(new T.BoxGeometry(0.32, 2.9, 0.32)), frameM)
    p.position.set(0, 1.45, sd * 1.18)
    p.castShadow = true
    gate.add(p)
  })
  const beam = new T.Mesh(keep(new T.BoxGeometry(0.6, 0.36, 2.7)), frameM)
  beam.position.set(0, 2.9, 0)
  beam.castShadow = true
  gate.add(beam)
  const stripM = keep(new T.MeshBasicMaterial({ color: 0x2fd3ea }))
  for (const a of [
    [0, 2.71, 0, 0.62, 0.05, 2.36],
    [0.17, 1.45, 1.18, 0.03, 2.4, 0.05],
    [0.17, 1.45, -1.18, 0.03, 2.4, 0.05],
    [-0.17, 1.45, 1.18, 0.03, 2.4, 0.05],
    [-0.17, 1.45, -1.18, 0.03, 2.4, 0.05],
  ]) {
    const m = new T.Mesh(keep(new T.BoxGeometry(a[3], a[4], a[5])), stripM)
    m.position.set(a[0], a[1], a[2])
    gate.add(m)
  }
  const gateSign = sprite('SCANNER GATE 01', 0x2fd3ea)
  gateSign.position.set(0, 3.45, 0)
  gateSign.scale.set(2.6, 0.33, 1)
  gate.add(gateSign)
  const curtainMat = keep(
    new T.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: T.AdditiveBlending,
      side: T.DoubleSide,
      uniforms: { uT: { value: 0 }, uC: { value: new T.Color(0x2fd3ea) }, uI: { value: 0.35 } },
      vertexShader: 'varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
      fragmentShader:
        'uniform float uT;uniform vec3 uC;uniform float uI;varying vec2 vUv;void main(){float lines=smoothstep(.92,1.,sin(vUv.y*220.+uT*6.)*.5+.5);float sweep=exp(-pow((vUv.y-fract(uT*.55))*14.,2.));float edge=smoothstep(0.,.08,vUv.x)*smoothstep(1.,.92,vUv.x);float a=(lines*.35+sweep*1.4+.05)*edge*uI;gl_FragColor=vec4(uC*a*1.6,a);}',
    }),
  )
  const curtain = new T.Mesh(keep(new T.PlaneGeometry(2.2, 2.5)), curtainMat)
  curtain.rotation.y = Math.PI / 2
  curtain.position.set(0, BELT_Y + 1.05, 0)
  gate.add(curtain)
  const fanMat = keep(new T.LineBasicMaterial({ color: 0x2fd3ea, transparent: true, opacity: 0.5, blending: T.AdditiveBlending }))
  const fp: number[] = []
  for (let i = 0; i <= 16; i++) fp.push(0, 2.7, 0, 0, BELT_Y + 0.02, -1 + (i * 2) / 16)
  const fanG = keep(new T.BufferGeometry())
  fanG.setAttribute('position', new T.Float32BufferAttribute(fp, 3))
  gate.add(new T.LineSegments(fanG, fanMat))

  // Gate read-out panel
  const pc = document.createElement('canvas')
  pc.width = 512
  pc.height = 256
  const px = pc.getContext('2d')!
  const pt = keep(new T.CanvasTexture(pc))
  pt.colorSpace = T.SRGBColorSpace
  const panel = new T.Mesh(keep(new T.PlaneGeometry(1.6, 0.8)), keep(new T.MeshBasicMaterial({ map: pt, transparent: true })))
  panel.position.set(0.32, 2.1, 1.36)
  panel.rotation.y = 0.35
  gate.add(panel)
  const drawPanel = (l: string[], col: string) => {
    px.clearRect(0, 0, 512, 256)
    px.fillStyle = 'rgba(8,10,14,.92)'
    px.fillRect(0, 0, 512, 256)
    px.strokeStyle = col
    px.lineWidth = 4
    px.strokeRect(2, 2, 508, 252)
    px.fillStyle = '#8c95a3'
    px.font = '500 22px "IBM Plex Mono", monospace'
    px.fillText(l[0], 22, 42)
    px.fillStyle = col
    px.font = '800 84px "Big Shoulders Display", sans-serif'
    px.fillText(l[1], 20, 140)
    px.fillStyle = '#d9dee6'
    px.font = '500 22px "IBM Plex Mono", monospace'
    px.fillText(l[2], 22, 196)
    px.fillText(l[3] ?? '', 22, 232)
    pt.needsUpdate = true
  }
  drawPanel(['GATE 01 · READY', 'STANDBY', 'booking-time check', 'before the label prints'], '#2fd3ea')

  // Parcels: kraft boxes with a white label on top
  const kraft = (sd: number) =>
    canvasTex(256, 256, (x) => {
      x.fillStyle = `hsl(${24 + sd * 10},40%,${42 + sd * 8}%)`
      x.fillRect(0, 0, 256, 256)
      for (let i = 0; i < 2600; i++) {
        x.fillStyle = `rgba(${Math.random() < 0.5 ? '60,35,15' : '240,210,170'},${Math.random() * 0.09})`
        x.fillRect(Math.random() * 256, Math.random() * 256, 1 + Math.random() * 3, 1)
      }
      x.fillStyle = 'rgba(215,190,150,.65)'
      x.fillRect(104, 0, 48, 256)
    })
  const kTex = [kraft(0), kraft(0.5), kraft(1)]
  const kMat = kTex.map((t) => keep(new T.MeshStandardMaterial({ map: t, roughness: 0.9 })))
  const lblTex = canvasTex(128, 160, (x) => {
    x.fillStyle = '#f6f6f2'
    x.fillRect(0, 0, 128, 160)
    x.fillStyle = '#111'
    x.fillRect(10, 12, 60, 7)
    x.fillRect(10, 26, 40, 5)
    x.font = '900 42px sans-serif'
    x.fillText('SP', 10, 86)
    for (let i = 0; i < 26; i++) x.fillRect(12 + i * 4, 104, 1 + ((i * 7) % 3), 40)
  })
  const boxG = keep(new T.BoxGeometry(1, 1, 1))
  const lblG = keep(new T.PlaneGeometry(1, 1))
  lblG.rotateX(-Math.PI / 2)
  const lblM = keep(new T.MeshStandardMaterial({ map: lblTex, roughness: 0.6 }))
  const tagG = keep(new T.BoxGeometry(0.18, 0.06, 0.18))

  const parcels: THREE.Group[] = []
  const bay: THREE.Group[] = []
  const N = 24
  const pd = (g: THREE.Group) => g.userData as ParcelData
  const newParcel = () => {
    const g = new T.Group()
    const w = 0.45 + Math.random() * 0.5
    const d = 0.4 + Math.random() * 0.45
    const h = 0.25 + Math.random() * 0.42
    const m = new T.Mesh(boxG, kMat[Math.floor(Math.random() * 3)])
    m.scale.set(w, h, d)
    m.position.y = h / 2
    m.castShadow = true
    m.receiveShadow = true
    g.add(m)
    const l = new T.Mesh(lblG, lblM)
    l.scale.set(w * 0.5, 1, Math.min(d * 0.6, w * 0.62))
    l.position.set(-w * 0.12, h + 0.003, 0)
    g.add(l)
    const tag = new T.Mesh(tagG, keep(new T.MeshBasicMaterial({ color: 0xffffff })))
    tag.position.set(w * 0.3, h + 0.05, d * 0.25)
    tag.visible = false
    g.add(tag)
    const data: ParcelData = {
      w,
      h,
      d,
      box: m,
      tag,
      state: 'belt',
      acc: Math.random().toString(16).slice(2, 10),
      dest: `${rnd(STATES)} ${100 + Math.floor(Math.random() * 899)}`,
      act: null,
      t: 0,
      from: new T.Vector3(),
      bayIdx: -1,
      phase: Math.random() * 6,
    }
    g.userData = data
    scene.add(g)
    return g
  }
  for (let i = 0; i < N; i++) {
    const p = newParcel()
    p.position.set(X0 + i * ((X1 - X0) / N), BELT_Y, (Math.random() - 0.5) * 0.5)
    if (p.position.x > 0) pd(p).act = 'allow'
    parcels.push(p)
  }
  const decide = (): Act => {
    const r = Math.random()
    return r < 0.72 ? 'allow' : r < 0.79 ? 'scan' : r < 0.84 ? 'owner' : r < 0.89 ? 'review' : r < 0.96 ? 'hold' : 'block'
  }

  // Dust in the lamp light
  const DN = 700
  const dG = keep(new T.BufferGeometry())
  const dP = new Float32Array(DN * 3)
  const dV = new Float32Array(DN)
  for (let i = 0; i < DN; i++) {
    dP[i * 3] = (Math.random() - 0.5) * 22
    dP[i * 3 + 1] = Math.random() * 8
    dP[i * 3 + 2] = (Math.random() - 0.5) * 10
    dV[i] = 0.05 + Math.random() * 0.12
  }
  dG.setAttribute('position', new T.BufferAttribute(dP, 3))
  scene.add(new T.Points(dG, keep(new T.PointsMaterial({ color: 0xfff0dc, size: 0.035, transparent: true, opacity: 0.55, depthWrite: false, blending: T.AdditiveBlending }))))

  // Bloom
  const comp = new EffectComposer(R)
  comp.addPass(new RenderPass(scene, cam))
  const bloom = new UnrealBloomPass(new T.Vector2(512, 512), 0.62, 0.5, 0.7)
  comp.addPass(bloom)
  comp.addPass(new OutputPass())

  // Hover read-out
  const ray = new T.Raycaster()
  const mouse = new T.Vector2()
  const onMove = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect()
    mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1
    mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1
    ray.setFromCamera(mouse, cam)
    const hit = ray.intersectObjects(parcels.map((p) => pd(p).box))[0]
    if (hit) {
      const u = pd(hit.object.parent as THREE.Group)
      tip.style.display = 'block'
      tip.style.left = `${e.clientX - r.left}px`
      tip.style.top = `${e.clientY - r.top}px`
      tip.innerHTML = `<b>acct ${u.acc}</b> → ${u.dest}<br>${u.act ? (u.act === 'allow' ? 'Allowed · label printed' : `${NAME[u.act]} · pulled off the line`) : 'Not scanned yet'}`
      canvas.style.cursor = 'pointer'
    } else {
      tip.style.display = 'none'
      canvas.style.cursor = 'grab'
    }
  }
  const onLeave = () => {
    tip.style.display = 'none'
  }
  const onDown = () => {
    ctl.autoRotate = false
  }
  canvas.addEventListener('pointermove', onMove)
  canvas.addEventListener('pointerleave', onLeave)
  canvas.addEventListener('pointerdown', onDown)

  const resize = () => {
    const r = host.getBoundingClientRect()
    if (!r.width || !r.height) return
    R.setSize(r.width, r.height, false)
    cam.aspect = r.width / r.height
    cam.fov = r.width < 700 ? 52 : 36
    cam.updateProjectionMatrix()
    comp.setSize(r.width, r.height)
    bloom.resolution.set(r.width, r.height)
  }
  const ro = new ResizeObserver(resize)
  ro.observe(host)
  resize()

  const clock = new T.Clock()
  const SPEED = reduce ? 0.6 : 1.5
  const gateCol = new T.Color(0x2fd3ea)
  const base = new T.Color(0x2fd3ea)
  let gateFlash = 0
  let armOpen = 0
  const bayPos = (i: number) => {
    const a = i * 2.39996
    const rr = 0.35 + Math.sqrt(i) * 0.42
    return new T.Vector3(BAY.x + Math.cos(a) * rr, 0, BAY.z + Math.sin(a) * rr)
  }
  const resetP = (p: THREE.Group) => {
    const u = pd(p)
    let minX = Infinity
    for (const q of parcels) if (q !== p && pd(q).state === 'belt') minX = Math.min(minX, q.position.x)
    p.position.set(Math.min(X0 + 0.3, minX - 1.25), BELT_Y, (Math.random() - 0.5) * 0.5)
    p.rotation.set(0, (Math.random() - 0.5) * 0.3, 0)
    u.state = 'belt'
    u.act = null
    u.tag.visible = false
    u.acc = Math.random().toString(16).slice(2, 10)
    u.dest = `${rnd(STATES)} ${100 + Math.floor(Math.random() * 899)}`
  }

  let raf = 0
  const frame = () => {
    raf = requestAnimationFrame(frame)
    const dt = Math.min(0.05, clock.getDelta())
    const t = clock.elapsedTime
    bt.offset.x -= (dt * SPEED * 8) / (X1 - X0)
    curtainMat.uniforms.uT.value = t
    gateFlash = Math.max(0, gateFlash - dt * 1.4)
    curtainMat.uniforms.uI.value = 0.32 + gateFlash * 1.4
    ;(curtainMat.uniforms.uC.value as THREE.Color).lerp(gateCol, 0.12)
    stripM.color.lerp(gateFlash > 0.05 ? gateCol : base, 0.1)
    fanMat.color.copy(stripM.color)
    fanMat.opacity = 0.25 + gateFlash * 0.6
    armOpen = Math.max(0, armOpen - dt * 1.6)
    armPivot.rotation.y = -Math.min(1, armOpen) * 0.85
    bayRingMat.opacity = 0.35 + Math.sin(t * 2.2) * 0.15
    for (const p of parcels) {
      const u = pd(p)
      if (u.state === 'belt') {
        p.position.x += SPEED * dt
        if (u.act === null && p.position.x >= 0) {
          u.act = decide()
          gateFlash = 1
          gateCol.setHex(HEX[u.act])
          if (u.act !== 'allow') {
            u.tag.visible = true
            ;(u.tag.material as THREE.MeshBasicMaterial).color.setHex(HEX[u.act])
          }
          drawPanel([`GATE 01 · acct ${u.acc}`, NAME[u.act].toUpperCase(), `→ ${u.dest}`, `decided in ${86 + Math.floor(Math.random() * 16)} ms`], hexs(HEX[u.act]))
        }
        if (u.act && u.act !== 'allow' && p.position.x >= DIV_X - 0.4) {
          u.state = 'chute'
          u.t = 0
          armOpen = 1.4
          u.from = p.position.clone()
          u.bayIdx = bay.length
          bay.push(p)
          if (bay.length > 16) {
            resetP(bay.shift()!)
            bay.forEach((b, i) => (pd(b).bayIdx = i))
          }
        }
        if (p.position.x > X1 - 0.3) resetP(p)
      } else if (u.state === 'chute') {
        u.t += dt * 0.9
        const k = Math.min(1, u.t)
        const e = k * k * (3 - 2 * k)
        const to = bayPos(u.bayIdx)
        p.position.set(u.from.x + (to.x - u.from.x) * e, BELT_Y * (1 - e) + Math.sin(k * Math.PI) * 0.35, u.from.z + (to.z - u.from.z) * e)
        p.rotation.y = e * 1.2
        if (k >= 1) u.state = 'bay'
      } else {
        const to = bayPos(u.bayIdx)
        p.position.x += (to.x - p.position.x) * 0.08
        p.position.z += (to.z - p.position.z) * 0.08
        p.position.y = 0
      }
      u.tag.position.y = u.h + 0.05 + Math.sin(t * 4 + u.phase) * 0.02
    }
    if (!reduce) {
      const a = dG.attributes.position.array as Float32Array
      for (let i = 0; i < DN; i++) {
        a[i * 3 + 1] += dV[i] * dt * 0.3
        if (a[i * 3 + 1] > 8) a[i * 3 + 1] = 0
      }
      dG.attributes.position.needsUpdate = true
    }
    ctl.update()
    comp.render()
  }
  frame()
  // Repaint the gate panel once the display face has loaded.
  document.fonts?.ready.then(() => drawPanel(['GATE 01 · READY', 'STANDBY', 'booking-time check', 'before the label prints'], '#2fd3ea'))

  return {
    dispose() {
      cancelAnimationFrame(raf)
      ro.disconnect()
      canvas.removeEventListener('pointermove', onMove)
      canvas.removeEventListener('pointerleave', onLeave)
      canvas.removeEventListener('pointerdown', onDown)
      ctl.dispose()
      comp.dispose()
      disposables.forEach((d) => d.dispose())
      R.dispose()
    },
  }
}
