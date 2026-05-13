import { useState } from "react";
import Sidebar from "../components/Sidebar";

const PRESETS = [
  { name: "HVAC", apiName: "HVAC", type: "continuous", power: 2.5, duration: null, deadline: null, comfort: true, lighting: false },
  { name: "Aydınlatma", apiName: "Lighting", type: "continuous", power: 0.2, duration: null, deadline: null, comfort: false, lighting: true },
  { name: "Çamaşır Makinesi", apiName: "Washing Machine", type: "shiftable", power: 1.5, duration: 2, deadline: 22, comfort: false, lighting: false },
  { name: "Bulaşık Makinesi", apiName: "Dishwasher", type: "shiftable", power: 1.5, duration: 2, deadline: 23, comfort: false, lighting: false },
  { name: "Su Isıtıcı", apiName: "Water Heater", type: "shiftable", power: 1.5, duration: 1, deadline: 22, comfort: false, lighting: false },
  { name: "EV Şarj", apiName: "EV Charger", type: "shiftable", power: 7.2, duration: 3, deadline: 8, comfort: false, lighting: false },
  { name: "Çamaşır Kurutma", apiName: "Tumble Dryer", type: "shiftable", power: 2.5, duration: 2, deadline: 22, comfort: false, lighting: false },
];

const COLORS = [
  { bg: "rgba(29,158,117,0.15)", color: "#5DCAA5" },
  { bg: "rgba(55,138,221,0.15)", color: "#85B7EB" },
  { bg: "rgba(250,199,117,0.15)", color: "#FAC775" },
  { bg: "rgba(168,85,247,0.15)", color: "#c084fc" },
  { bg: "rgba(239,159,39,0.15)", color: "#EF9F27" },
];

const EMPTY_FORM = { name: "", apiName: null, type: "continuous", power: "", duration: "", deadline: 22, comfort: false, lighting: false };

export default function DeviceSetup({ goTo, formData, updateForm }) {
  const [devices, setDevices] = useState(formData.devices || []);
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  const [limitMsg, setLimitMsg] = useState(false);

  const openModal = () => {
    if (devices.filter(d => d.apiName !== "HVAC" && d.apiName !== "Lighting").length >= 5) { setLimitMsg(true); setTimeout(() => setLimitMsg(false), 2500); return; }
    setForm(EMPTY_FORM); setShowModal(true);
  };
  const closeModal = () => setShowModal(false);
  const fillPreset = (p) => setForm({ name: p.name, apiName: p.apiName, type: p.type, power: p.power, duration: p.duration || "", deadline: p.deadline || 22, comfort: p.comfort, lighting: p.lighting });
  const saveDevice = () => {
    if (!form.name || !form.power) return;
    const newDevices = [...devices, { ...form, power: parseFloat(form.power) }];
    setDevices(newDevices);
    closeModal();
  };
  const removeDevice = (idx) => setDevices(devices.filter((_, i) => i !== idx));
  const toggleActive = (idx) => setDevices(devices.map((d, i) =>
    i === idx ? { ...d, activeToday: d.activeToday === false } : d
  ));
  const updateDeadline = (idx, val) => setDevices(devices.map((d, i) =>
    i === idx ? { ...d, deadline: val } : d
  ));
  const handleNext = () => {
    updateForm({ devices });
    goTo(devices.some(d => d.apiName === "HVAC") ? "comfort" : "summary");
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0a1628", display: "flex", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{
        position: "fixed", inset: 0,
        backgroundImage: "linear-gradient(rgba(29,158,117,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.05) 1px, transparent 1px)",
        backgroundSize: "40px 40px", pointerEvents: "none"
      }} />

      <Sidebar currentStep="devices" />

      <div style={{ flex: 1, padding: "2rem", position: "relative", zIndex: 1, display: "flex", flexDirection: "column" }}>

        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 26, color: "#f0f4f8", marginBottom: ".35rem" }}>
          Evindeki cihazları ekle
        </div>
        <p style={{ fontSize: 13, color: "rgba(240,244,248,0.65)", marginBottom: "1rem", fontWeight: 300 }}>
          Hangi cihazları kullanmak istiyorsan ekle.
        </p>

        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: "1rem", minHeight: 60 }}>
          {devices.length === 0 ? (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "2rem", border: "0.5px dashed rgba(29,158,117,0.25)", borderRadius: 12, gap: 8 }}>
              <div style={{ fontSize: 28, opacity: .4 }}>⚡</div>
              <div style={{ fontSize: 13, color: "rgba(240,244,248,0.4)" }}>Henüz cihaz eklenmedi</div>
            </div>
          ) : devices.map((d, i) => {
            const c = COLORS[i % COLORS.length];
            const initials = d.name.split(" ").map(w => w[0]).join("").substring(0, 2).toUpperCase();
            const isActive = d.activeToday !== false;
            return (
              <div key={i} style={{ background: isActive ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.02)", border: `0.5px solid ${isActive ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,0.05)"}`, borderLeft: `3px solid ${isActive ? c.color : "rgba(255,255,255,0.1)"}`, borderRadius: 10, padding: ".75rem 1rem", display: "flex", alignItems: "center", gap: 10, opacity: isActive ? 1 : 0.5, transition: "all .2s" }}>
                <div style={{ width: 34, height: 34, borderRadius: 9, background: c.bg, color: c.color, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 600, flexShrink: 0 }}>{initials}</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 500, color: "#f0f4f8", display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                    {d.name}
                    <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 4, fontWeight: 600, background: d.type === "shiftable" ? "rgba(55,138,221,0.18)" : "rgba(29,158,117,0.18)", color: d.type === "shiftable" ? "#85B7EB" : "#5DCAA5" }}>{d.type === "shiftable" ? "Ertelenebilir" : "Sürekli"}</span>
                    {!isActive && <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 4, background: "rgba(255,255,255,0.07)", color: "rgba(240,244,248,0.45)" }}>Bugün pasif</span>}
                  </div>
                  <div style={{ fontSize: 11, color: "rgba(240,244,248,0.5)", marginTop: 3, display: "flex", alignItems: "center", gap: 5 }}>
                    <span>{d.power} kW</span>
                    {d.type === "shiftable" && (
                      <>
                        <span>·</span>
                        <span>{d.duration} saat</span>
                        <span>·</span>
                        <span style={{ color: "rgba(240,244,248,0.4)" }}>deadline</span>
                        <select
                          value={d.deadline || 22}
                          onChange={e => updateDeadline(i, parseInt(e.target.value))}
                          style={{ fontSize: 11, background: "rgba(255,255,255,0.06)", border: "0.5px solid rgba(255,255,255,0.15)", borderRadius: 4, color: "#f0f4f8", padding: "1px 4px", cursor: "pointer", outline: "none" }}
                        >
                          {Array.from({ length: 24 }, (_, h) => <option key={h} value={h}>{String(h).padStart(2,"0")}:00</option>)}
                        </select>
                      </>
                    )}
                  </div>
                </div>
                <div onClick={() => toggleActive(i)} title={isActive ? "Bugün pasif yap" : "Bugün aktif et"} style={{ width: 36, height: 20, borderRadius: 10, cursor: "pointer", background: isActive ? "#1D9E75" : "rgba(255,255,255,0.12)", position: "relative", transition: "background .2s", flexShrink: 0 }}>
                  <div style={{ position: "absolute", width: 14, height: 14, borderRadius: "50%", background: "#fff", top: 3, left: isActive ? 19 : 3, transition: "left .2s", boxShadow: "0 1px 3px rgba(0,0,0,0.3)" }} />
                </div>
                <button onClick={() => removeDevice(i)} style={{ width: 26, height: 26, borderRadius: 6, border: "0.5px solid rgba(255,255,255,0.12)", background: "rgba(226,75,74,0.08)", color: "rgba(226,75,74,0.6)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15, flexShrink: 0 }}>×</button>
              </div>
            );
          })}
        </div>

        <button onClick={openModal} style={{ width: "100%", background: "rgba(29,158,117,0.1)", border: "1px dashed rgba(29,158,117,0.5)", borderRadius: 10, padding: 11, fontSize: 13, fontWeight: 500, color: "#5DCAA5", fontFamily: "'DM Sans', sans-serif", cursor: "pointer", marginBottom: limitMsg ? ".4rem" : "1rem" }}>
          + Cihaz Ekle
        </button>
        {limitMsg && (
          <div style={{ fontSize: 12, color: "#FAC775", background: "rgba(250,199,117,0.08)", border: "0.5px solid rgba(250,199,117,0.25)", borderRadius: 8, padding: "7px 12px", marginBottom: "1rem", textAlign: "center" }}>
            En fazla 5 cihaz eklenebilir.
          </div>
        )}

        <div style={{ marginTop: "auto", display: "flex", justifyContent: "space-between", paddingTop: "1rem" }}>
          <button onClick={() => goTo("mode")} style={{ background: "transparent", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "10px 20px", fontSize: 13, color: "rgba(240,244,248,0.45)", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>← Geri</button>
          <button onClick={handleNext} style={{ background: "#1D9E75", border: "none", borderRadius: 8, padding: "10px 28px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>Devam Et →</button>
        </div>
      </div>

      {showModal && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 }}>
          <div style={{ background: "#0f1e35", border: "0.5px solid rgba(255,255,255,0.14)", borderTop: "2px solid #1D9E75", borderRadius: 14, padding: "1.5rem", width: 340, maxWidth: "90vw" }}>
            <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 18, color: "#f0f4f8", marginBottom: "1.25rem" }}>Cihaz Ekle</div>

            <div style={{ marginBottom: ".75rem" }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: "#5DCAA5", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: 6 }}>Hazır şablondan seç</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                {PRESETS.map(p => (
                  <button key={p.name} onClick={() => fillPreset(p)} style={{ fontSize: 11, padding: "4px 11px", borderRadius: 20, border: "0.5px solid rgba(29,158,117,0.3)", background: "rgba(29,158,117,0.07)", color: "rgba(240,244,248,0.7)", cursor: "pointer", fontFamily: "'DM Sans', sans-serif" }}>{p.name}</button>
                ))}
              </div>
            </div>

            <div style={{ marginBottom: "1rem" }}>
              <label style={{ fontSize: 11, fontWeight: 500, color: "rgba(240,244,248,0.5)", letterSpacing: ".06em", textTransform: "uppercase", display: "block", marginBottom: 5 }}>Cihaz adı</label>
              <input type="text" placeholder="örn. Benim Klimam" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })}
                style={{ width: "100%", background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 13, color: "#f0f4f8", fontFamily: "'DM Sans', sans-serif", outline: "none" }} />
            </div>

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
                <input type="number" placeholder="örn. 1.5" value={form.power} onChange={e => setForm({ ...form, power: e.target.value })}
                  style={{ width: "100%", background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 13, color: "#f0f4f8", fontFamily: "'DM Sans', sans-serif", outline: "none" }} />
              </div>
            </div>

            {form.type === "shiftable" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: "1rem", paddingTop: ".75rem", borderTop: "0.5px solid rgba(255,255,255,0.07)" }}>
                <div>
                  <label style={{ fontSize: 11, fontWeight: 500, color: "rgba(240,244,248,0.5)", letterSpacing: ".06em", textTransform: "uppercase", display: "block", marginBottom: 5 }}>Süre (saat)</label>
                  <input type="number" placeholder="örn. 2" value={form.duration} onChange={e => setForm({ ...form, duration: e.target.value })}
                    style={{ width: "100%", background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 13, color: "#f0f4f8", fontFamily: "'DM Sans', sans-serif", outline: "none" }} />
                </div>
                <div>
                  <label style={{ fontSize: 11, fontWeight: 500, color: "rgba(240,244,248,0.5)", letterSpacing: ".06em", textTransform: "uppercase", display: "block", marginBottom: 5 }}>Deadline</label>
                  <select value={form.deadline} onChange={e => setForm({ ...form, deadline: parseInt(e.target.value) })}
                    style={{ width: "100%", background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "8px 12px", fontSize: 13, color: "#f0f4f8", fontFamily: "'DM Sans', sans-serif" }}>
                    {Array.from({ length: 24 }, (_, h) => <option key={h} value={h}>{String(h).padStart(2,"0")}:00</option>)}
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
                  <div onClick={() => setForm({ ...form, [f.key]: !form[f.key] })} style={{ width: 36, height: 20, borderRadius: 10, cursor: "pointer", background: form[f.key] ? "#1D9E75" : "rgba(255,255,255,0.1)", position: "relative", transition: "background .2s", flexShrink: 0 }}>
                    <div style={{ position: "absolute", width: 14, height: 14, borderRadius: "50%", background: "#fff", top: 3, left: form[f.key] ? 19 : 3, transition: "left .2s" }} />
                  </div>
                </div>
              ))}
            </div>

            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={closeModal} style={{ flex: 1, background: "transparent", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: 9, fontSize: 13, color: "rgba(240,244,248,0.5)", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>İptal</button>
              <button onClick={saveDevice} style={{ flex: 1, background: "#1D9E75", border: "none", borderRadius: 8, padding: 9, fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>Ekle</button>
            </div>
          </div>
        </div>
      )}
      <style>{`select option { background: #0a1628; }`}</style>
    </div>
  );
}