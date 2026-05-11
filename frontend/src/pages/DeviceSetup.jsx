import { useState } from "react";

const SIDEBAR_STEPS = [
  { key: "welcome", label: "Hoşgeldiniz" },
  { key: "schedule", label: "Kullanıcı programı" },
  { key: "comfort", label: "Konfor tercihleri" },
  { key: "devices", label: "Cihaz seçimi" },
  { key: "summary", label: "Özet & onayla" },
];

const PRESETS = [
  { name: "HVAC", type: "continuous", power: 2.5, comfort: true, lighting: false },
  { name: "Çamaşır Makinesi", type: "shiftable", power: 1.5, duration: 2, deadline: 22, comfort: false, lighting: false },
  { name: "Bulaşık Makinesi", type: "shiftable", power: 1.5, duration: 2, deadline: 23, comfort: false, lighting: false },
  { name: "Aydınlatma", type: "continuous", power: 0.3, comfort: false, lighting: true },
  { name: "Su Isıtıcı", type: "shiftable", power: 1.5, duration: 1, deadline: 22, comfort: false, lighting: false },
];

const COLORS = [
  { bg: "rgba(29,158,117,0.15)", color: "#5DCAA5" },
  { bg: "rgba(55,138,221,0.15)", color: "#85B7EB" },
  { bg: "rgba(250,199,117,0.15)", color: "#FAC775" },
  { bg: "rgba(168,85,247,0.15)", color: "#c084fc" },
  { bg: "rgba(239,159,39,0.15)", color: "#EF9F27" },
];

const EMPTY_FORM = { name: "", type: "continuous", power: "", duration: "", deadline: 22, comfort: false, lighting: false };

export default function DeviceSetup({ goTo, formData, updateForm }) {
  const [devices, setDevices] = useState(formData.devices || []);
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);

  const openModal = () => { setForm(EMPTY_FORM); setShowModal(true); };
  const closeModal = () => setShowModal(false);

  const fillPreset = (p) => setForm({
    name: p.name, type: p.type, power: p.power,
    duration: p.duration || "", deadline: p.deadline || 22,
    comfort: p.comfort, lighting: p.lighting
  });

  const saveDevice = () => {
    if (!form.name || !form.power) return;
    const newDevices = [...devices, { ...form, power: parseFloat(form.power) }];
    setDevices(newDevices);
    closeModal();
  };

  const removeDevice = (idx) => setDevices(devices.filter((_, i) => i !== idx));

  const handleNext = () => {
    updateForm({ devices });
    goTo("summary");
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0a1628", display: "flex", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{
        position: "fixed", inset: 0,
        backgroundImage: "linear-gradient(rgba(29,158,117,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.05) 1px, transparent 1px)",
        backgroundSize: "40px 40px", pointerEvents: "none"
      }} />

      {/* Sidebar */}
      <div style={{
        width: 200, flexShrink: 0, borderRight: "0.5px solid rgba(255,255,255,0.07)",
        padding: "1.5rem 1.25rem", display: "flex", flexDirection: "column",
        gap: ".25rem", position: "relative", zIndex: 1
      }}>
        <div style={{
          fontFamily: "'DM Serif Display', serif", fontSize: 14, color: "#5DCAA5",
          marginBottom: "1.5rem", display: "flex", alignItems: "center", gap: 6
        }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#1D9E75", animation: "blink 2s ease infinite" }} />
          SmartHome RL
        </div>
        {SIDEBAR_STEPS.map((step, i) => {
          const isDone = i < 3;
          const isActive = step.key === "devices";
          return (
            <div key={step.key} style={{
              display: "flex", alignItems: "center", gap: 10, padding: "8px 10px",
              borderRadius: 8, background: isActive ? "rgba(29,158,117,0.12)" : "transparent"
            }}>
              <div style={{
                width: 22, height: 22, borderRadius: "50%", flexShrink: 0,
                display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11,
                background: isDone ? "rgba(29,158,117,0.2)" : isActive ? "#1D9E75" : "transparent",
                border: isDone ? "0.5px solid #1D9E75" : isActive ? "none" : "0.5px solid rgba(255,255,255,0.15)",
                color: isDone ? "#5DCAA5" : isActive ? "#fff" : "rgba(240,244,248,0.35)"
              }}>
                {isDone ? "✓" : i + 1}
              </div>
              <span style={{
                fontSize: 12,
                color: isDone ? "#5DCAA5" : isActive ? "#f0f4f8" : "rgba(240,244,248,0.3)",
                fontWeight: isActive ? 500 : 400
              }}>{step.label}</span>
            </div>
          );
        })}
      </div>

      {/* Main */}
      <div style={{ flex: 1, padding: "2rem", position: "relative", zIndex: 1, display: "flex", flexDirection: "column" }}>
        <div style={{ height: 2, background: "rgba(255,255,255,0.07)", borderRadius: 2, marginBottom: "2rem", overflow: "hidden" }}>
          <div style={{ height: "100%", width: "84%", background: "#1D9E75", borderRadius: 2 }} />
        </div>

        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 26, color: "#f0f4f8", marginBottom: ".35rem" }}>
          Evindeki cihazları ekle
        </div>
        <p style={{ fontSize: 13, color: "rgba(240,244,248,0.4)", marginBottom: "1.5rem", fontWeight: 300 }}>
          Hangi cihazları kullanmak istiyorsan ekle. Maksimum 5 cihaz.
        </p>

        {/* Slot sayacı */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: "1rem" }}>
          <span style={{ fontSize: 12, color: "rgba(240,244,248,0.4)" }}>Eklenen cihaz:</span>
          <div style={{ display: "flex", gap: 4 }}>
            {[0,1,2,3,4].map(i => (
              <div key={i} style={{
                width: 24, height: 5, borderRadius: 3,
                background: i < devices.length ? "#1D9E75" : "rgba(255,255,255,0.08)",
                transition: "background .2s"
              }} />
            ))}
          </div>
          <span style={{ fontSize: 12, color: "rgba(240,244,248,0.4)" }}>{devices.length} / 5</span>
        </div>

        {/* Cihaz listesi */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: "1rem", minHeight: 60 }}>
          {devices.length === 0 ? (
            <div style={{
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
              padding: "1.5rem", border: "0.5px dashed rgba(255,255,255,0.12)", borderRadius: 10, gap: 6
            }}>
              <div style={{ fontSize: 24, opacity: .3 }}>○</div>
              <div style={{ fontSize: 13, color: "rgba(240,244,248,0.3)" }}>Henüz cihaz eklenmedi</div>
            </div>
          ) : devices.map((d, i) => {
            const c = COLORS[i % COLORS.length];
            const initials = d.name.split(" ").map(w => w[0]).join("").substring(0, 2).toUpperCase();
            const meta = `${d.power} kW · ${d.type === "shiftable" ? "ertelenebilir" : "sürekli"}${d.type === "shiftable" ? ` · ${d.duration} saat · deadline ${d.deadline}:00` : ""}`;
            return (
              <div key={i} style={{
                background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)",
                borderRadius: 10, padding: ".75rem 1rem", display: "flex", alignItems: "center", gap: 10
              }}>
                <div style={{
                  width: 32, height: 32, borderRadius: 8, background: c.bg, color: c.color,
                  display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 500, flexShrink: 0
                }}>{initials}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 500, color: "#f0f4f8" }}>
                    {d.name}
                    <span style={{
                      fontSize: 10, padding: "2px 7px", borderRadius: 4, fontWeight: 500, marginLeft: 8,
                      background: d.type === "shiftable" ? "rgba(55,138,221,0.15)" : "rgba(29,158,117,0.15)",
                      color: d.type === "shiftable" ? "#85B7EB" : "#5DCAA5"
                    }}>{d.type === "shiftable" ? "Ertelenebilir" : "Sürekli"}</span>
                  </div>
                  <div style={{ fontSize: 11, color: "rgba(240,244,248,0.4)", marginTop: 1 }}>{meta}</div>
                </div>
                <button onClick={() => removeDevice(i)} style={{
                  width: 24, height: 24, borderRadius: 6, border: "0.5px solid rgba(255,255,255,0.1)",
                  background: "transparent", color: "rgba(240,244,248,0.35)", cursor: "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14
                }}>×</button>
              </div>
            );
          })}
        </div>

        {/* Ekle butonu */}
        <button
          onClick={openModal}
          disabled={devices.length >= 5}
          style={{
            width: "100%", background: "rgba(29,158,117,0.1)",
            border: "0.5px dashed rgba(29,158,117,0.4)", borderRadius: 10, padding: 10,
            fontSize: 13, color: "#5DCAA5", fontFamily: "'DM Sans', sans-serif",
            cursor: devices.length >= 5 ? "not-allowed" : "pointer",
            opacity: devices.length >= 5 ? .35 : 1, marginBottom: "1rem"
          }}
        >+ Cihaz Ekle</button>

        {/* Footer */}
        <div style={{ marginTop: "auto", display: "flex", justifyContent: "space-between", paddingTop: "1rem" }}>
          <button onClick={() => goTo("comfort")} style={{
            background: "transparent", border: "0.5px solid rgba(255,255,255,0.12)",
            borderRadius: 8, padding: "10px 20px", fontSize: 13,
            color: "rgba(240,244,248,0.45)", fontFamily: "'DM Sans', sans-serif", cursor: "pointer"
          }}>← Geri</button>
          <button onClick={handleNext} style={{
            background: "#1D9E75", border: "none", borderRadius: 8,
            padding: "10px 28px", fontSize: 13, fontWeight: 500,
            color: "#fff", fontFamily: "'DM Sans', sans-serif", cursor: "pointer"
          }}>Devam Et →</button>
        </div>
      </div>

      {/* Modal */}
      {showModal && (
        <div style={{
          position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
          display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100
        }}>
          <div style={{
            background: "#0f1e35", border: "0.5px solid rgba(255,255,255,0.12)",
            borderRadius: 14, padding: "1.5rem", width: 340, maxWidth: "90vw"
          }}>
            <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 18, color: "#f0f4f8", marginBottom: "1.25rem" }}>
              Cihaz Ekle
            </div>

            {/* Presetler */}
            <div style={{ marginBottom: ".75rem" }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: "rgba(240,244,248,0.5)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: 5 }}>
                Hazır şablondan seç
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                {PRESETS.map(p => (
                  <button key={p.name} onClick={() => fillPreset(p)} style={{
                    fontSize: 11, padding: "3px 10px", borderRadius: 20,
                    border: "0.5px solid rgba(255,255,255,0.12)", background: "transparent",
                    color: "rgba(240,244,248,0.5)", cursor: "pointer", fontFamily: "'DM Sans', sans-serif"
                  }}>{p.name}</button>
                ))}
              </div>
            </div>

            {/* Form */}
            {[
              { label: "Cihaz adı", type: "text", key: "name", placeholder: "örn. Benim Klimam" },
            ].map(f => (
              <div key={f.key} style={{ marginBottom: "1rem" }}>
                <label style={{ fontSize: 11, fontWeight: 500, color: "rgba(240,244,248,0.5)", letterSpacing: ".06em", textTransform: "uppercase", display: "block", marginBottom: 5 }}>{f.label}</label>
                <input
                  type={f.type} placeholder={f.placeholder} value={form[f.key]}
                  onChange={e => setForm({ ...form, [f.key]: e.target.value })}
                  style={{
                    width: "100%", background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.12)",
                    borderRadius: 8, padding: "8px 12px", fontSize: 13, color: "#f0f4f8",
                    fontFamily: "'DM Sans', sans-serif", outline: "none"
                  }}
                />
              </div>
            ))}

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: "1rem" }}>
              <div>
                <label style={{ fontSize: 11, fontWeight: 500, color: "rgba(240,244,248,0.5)", letterSpacing: ".06em", textTransform: "uppercase", display: "block", marginBottom: 5 }}>Cihaz tipi</label>
                <select value={form.type} onChange={e => setForm({ ...form, type: e.target.value })}
                  style={{ width: "100%", background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 13, color: "#f0f4f8", fontFamily: "'DM Sans', sans-serif" }}>
                  <option value="continuous">Sürekli</option>
                  <option value="shiftable">Ertelenebilir</option>
                </select>
              </div>
              <div>
                <label style={{ fontSize: 11, fontWeight: 500, color: "rgba(240,244,248,0.5)", letterSpacing: ".06em", textTransform: "uppercase", display: "block", marginBottom: 5 }}>Güç (kW)</label>
                <input type="number" placeholder="örn. 1.5" value={form.power}
                  onChange={e => setForm({ ...form, power: e.target.value })}
                  style={{ width: "100%", background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 13, color: "#f0f4f8", fontFamily: "'DM Sans', sans-serif", outline: "none" }}
                />
              </div>
            </div>

            {form.type === "shiftable" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: "1rem", paddingTop: ".75rem", borderTop: "0.5px solid rgba(255,255,255,0.07)" }}>
                <div>
                  <label style={{ fontSize: 11, fontWeight: 500, color: "rgba(240,244,248,0.5)", letterSpacing: ".06em", textTransform: "uppercase", display: "block", marginBottom: 5 }}>Süre (saat)</label>
                  <input type="number" placeholder="örn. 2" value={form.duration}
                    onChange={e => setForm({ ...form, duration: e.target.value })}
                    style={{ width: "100%", background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 13, color: "#f0f4f8", fontFamily: "'DM Sans', sans-serif", outline: "none" }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 11, fontWeight: 500, color: "rgba(240,244,248,0.5)", letterSpacing: ".06em", textTransform: "uppercase", display: "block", marginBottom: 5 }}>Deadline</label>
                  <select value={form.deadline} onChange={e => setForm({ ...form, deadline: parseInt(e.target.value) })}
                    style={{ width: "100%", background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 13, color: "#f0f4f8", fontFamily: "'DM Sans', sans-serif" }}>
                    {[18,19,20,21,22,23].map(h => <option key={h} value={h}>{h}:00</option>)}
                  </select>
                </div>
              </div>
            )}

            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: "1.25rem" }}>
              {[
                { key: "comfort", label: "Sıcaklık konforunu etkiler mi? (HVAC gibi)" },
                { key: "lighting", label: "Aydınlatma etkili mi?" },
              ].map(f => (
                <div key={f.key} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontSize: 12, color: "rgba(240,244,248,0.6)", flex: 1 }}>{f.label}</span>
                  <div onClick={() => setForm({ ...form, [f.key]: !form[f.key] })} style={{
                    width: 36, height: 20, borderRadius: 10, cursor: "pointer",
                    background: form[f.key] ? "#1D9E75" : "rgba(255,255,255,0.1)",
                    position: "relative", transition: "background .2s", flexShrink: 0
                  }}>
                    <div style={{
                      position: "absolute", width: 14, height: 14, borderRadius: "50%",
                      background: "#fff", top: 3, left: form[f.key] ? 19 : 3, transition: "left .2s"
                    }} />
                  </div>
                </div>
              ))}
            </div>

            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={closeModal} style={{
                flex: 1, background: "transparent", border: "0.5px solid rgba(255,255,255,0.12)",
                borderRadius: 8, padding: 9, fontSize: 13, color: "rgba(240,244,248,0.5)",
                fontFamily: "'DM Sans', sans-serif", cursor: "pointer"
              }}>İptal</button>
              <button onClick={saveDevice} style={{
                flex: 1, background: "#1D9E75", border: "none", borderRadius: 8,
                padding: 9, fontSize: 13, fontWeight: 500, color: "#fff",
                fontFamily: "'DM Sans', sans-serif", cursor: "pointer"
              }}>Ekle</button>
            </div>
          </div>
        </div>
      )}

      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} } select option { background: #0a1628; }`}</style>
    </div>
  );
}