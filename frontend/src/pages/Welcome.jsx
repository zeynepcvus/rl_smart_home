import { useEffect, useState } from "react";

const DEVICES = [
  { icon: "AC", bg: "rgba(29,158,117,0.15)", color: "#5DCAA5", name: "HVAC", desc: "akıllı ısı kontrolü" },
  { icon: "WM", bg: "rgba(55,138,221,0.15)", color: "#85B7EB", name: "Çamaşır", desc: "ucuz saate kaydır" },
  { icon: "DW", bg: "rgba(168,85,247,0.15)", color: "#c084fc", name: "Bulaşık", desc: "gece çalıştır" },
  { icon: "LT", bg: "rgba(250,199,117,0.15)", color: "#FAC775", name: "Aydınlatma", desc: "ihtiyaca göre" },
  { icon: "WH", bg: "rgba(239,159,39,0.15)", color: "#EF9F27", name: "Su Isıtıcı", desc: "enerji verimli" },
];

const STEPS = [
  { num: "1", title: "Cihazlarını tanıt", desc: "Hangi cihazları kullandığını ve günlük programını belirt." },
  { num: "2", title: "Mod seç", desc: "Maliyet, konfor veya dengeli optimizasyon modlarından birini seç." },
  { num: "3", title: "Sonucu gör", desc: "RL ajanı 24 saatlik simülasyon çalıştırır, kural tabanlıyla karşılaştırır." },
];

export default function Welcome({ goTo, hasSavedProfile, loadProfile, clearProfile }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => setVisible(true), []);

  return (
    <div style={{
      minHeight: "100vh", background: "#0a1628", display: "flex",
      flexDirection: "column", alignItems: "center", justifyContent: "center",
      padding: "3rem 2rem", position: "relative", overflow: "hidden",
      fontFamily: "'DM Sans', sans-serif"
    }}>
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage: "linear-gradient(rgba(29,158,117,0.07) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.07) 1px, transparent 1px)",
        backgroundSize: "40px 40px", pointerEvents: "none"
      }} />
      <div style={{
        position: "absolute", width: 320, height: 320, borderRadius: "50%",
        background: "radial-gradient(circle, rgba(29,158,117,0.18) 0%, transparent 70%)",
        top: "50%", left: "50%", transform: "translate(-50%, -60%)", pointerEvents: "none"
      }} />

      {/* Badge */}
      <div style={{
        display: "inline-flex", alignItems: "center", gap: 6,
        background: "rgba(29,158,117,0.12)", border: "0.5px solid rgba(29,158,117,0.35)",
        borderRadius: 20, padding: "5px 14px", fontSize: 11, color: "#5DCAA5",
        letterSpacing: ".08em", textTransform: "uppercase", fontWeight: 500,
        marginBottom: "1.75rem", position: "relative",
        opacity: visible ? 1 : 0, transition: "opacity 0.6s ease"
      }}>
        <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#1D9E75", animation: "blink 2s ease infinite" }} />
        Reinforcement Learning · PPO
      </div>

      {/* Başlık */}
      <h1 style={{
        fontFamily: "'DM Serif Display', serif", fontSize: 48, color: "#f0f4f8",
        textAlign: "center", lineHeight: 1.1, marginBottom: ".5rem",
        position: "relative", opacity: visible ? 1 : 0, transition: "opacity 0.7s ease 0.1s"
      }}>
        Akıllı Evin<br />
        <em style={{ fontStyle: "italic", color: "#5DCAA5" }}>Enerji Yöneticisi</em>
      </h1>

      {/* Alt başlık */}
      <p style={{
        fontSize: 15, color: "rgba(240,244,248,0.45)", textAlign: "center",
        maxWidth: 400, lineHeight: 1.7, marginBottom: "2.5rem", fontWeight: 300,
        position: "relative", opacity: visible ? 1 : 0, transition: "opacity 0.7s ease 0.2s"
      }}>
        Cihazlarını tanıt, konfor tercihlerini belirle —<br />
        sistem optimumu senin için hesaplar.
      </p>

      {/* Buton */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, opacity: visible ? 1 : 0, transition: "opacity 0.7s ease 0.3s" }}>
        {hasSavedProfile && (
          <div style={{ display: "flex", alignItems: "center", gap: 6, background: "rgba(29,158,117,0.1)", border: "0.5px solid rgba(29,158,117,0.3)", borderRadius: 20, padding: "4px 12px", fontSize: 11, color: "#5DCAA5" }}>
            <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#1D9E75" }} />
            Kayıtlı ev profili bulundu
          </div>
        )}
        <div style={{ display: "flex", gap: 10 }}>
          {hasSavedProfile && (
            <button
              onClick={() => { loadProfile(); goTo("summary"); }}
              style={{ background: "#1D9E75", color: "#fff", border: "none", borderRadius: 10, padding: "13px 28px", fontSize: 14, fontWeight: 500, cursor: "pointer", display: "flex", alignItems: "center", gap: 8 }}
              onMouseEnter={e => e.currentTarget.style.background = "#0F6E56"}
              onMouseLeave={e => e.currentTarget.style.background = "#1D9E75"}
            >
              Kayıtlı Evden Devam Et
              <span style={{ width: 18, height: 18, borderRadius: "50%", background: "rgba(255,255,255,0.2)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11 }}>→</span>
            </button>
          )}
          <button
            onClick={() => { if (hasSavedProfile) clearProfile(); goTo("schedule"); }}
            style={{
              background: hasSavedProfile ? "transparent" : "#1D9E75",
              color: hasSavedProfile ? "rgba(240,244,248,0.5)" : "#fff",
              border: hasSavedProfile ? "0.5px solid rgba(255,255,255,0.12)" : "none",
              borderRadius: 10, padding: "13px 28px", fontSize: 14, fontWeight: 500, cursor: "pointer",
              display: "flex", alignItems: "center", gap: 8
            }}
          >
            {hasSavedProfile ? "Yeni Ev Kur" : "Kuruluma Başla"}
            {!hasSavedProfile && (
              <span style={{ width: 18, height: 18, borderRadius: "50%", background: "rgba(255,255,255,0.2)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11 }}>→</span>
            )}
          </button>
        </div>
      </div>

      {/* İstatistikler */}
      <div style={{
        display: "flex", gap: "2rem", marginTop: "3rem", position: "relative",
        opacity: visible ? 1 : 0, transition: "opacity 0.7s ease 0.4s"
      }}>
        {[
          { val: "%10–90", label: "maliyet tasarrufu" },
          { val: "%75+", label: "daha az konfor ihlali" },
          { val: "3 mod", label: "maliyet · dengeli · konfor" },
        ].map((s, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: "2rem" }}>
            {i > 0 && <div style={{ width: 1, background: "rgba(240,244,248,0.1)", alignSelf: "stretch" }} />}
            <div style={{ textAlign: "center" }}>
              <span style={{ fontFamily: "'DM Serif Display', serif", fontSize: 22, color: "#5DCAA5", display: "block" }}>{s.val}</span>
              <span style={{ fontSize: 11, color: "rgba(240,244,248,0.35)", display: "block", marginTop: 2 }}>{s.label}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Nasıl çalışır */}
      <div style={{
        display: "flex", gap: 10, marginTop: "2.5rem", position: "relative",
        opacity: visible ? 1 : 0, transition: "opacity 0.7s ease 0.5s", maxWidth: 640, width: "100%",
        flexDirection: "column"
      }}>
        <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.3)", letterSpacing: ".08em", textTransform: "uppercase", textAlign: "center" }}>
          Nasıl çalışır?
        </div>
        <div style={{ display: "flex", gap: 10 }}>
        {STEPS.map((s, i) => (
          <div key={i} style={{
            flex: 1, background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.07)",
            borderRadius: 10, padding: "10px 12px", display: "flex", gap: 10, alignItems: "flex-start"
          }}>
            <div style={{
              width: 20, height: 20, borderRadius: "50%", background: "rgba(29,158,117,0.15)",
              color: "#5DCAA5", fontSize: 10, fontWeight: 600, display: "flex",
              alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 1
            }}>{s.num}</div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 500, color: "#f0f4f8", marginBottom: 2 }}>{s.title}</div>
              <div style={{ fontSize: 11, color: "rgba(240,244,248,0.4)", lineHeight: 1.5 }}>{s.desc}</div>
            </div>
          </div>
        ))}
        </div>
      </div>

      {/* Cihaz kartları */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginTop: "1.25rem",
        position: "relative", opacity: visible ? 1 : 0, transition: "opacity 0.7s ease 0.6s", maxWidth: 640, width: "100%"
      }}>
        {DEVICES.map((c, i) => (
          <div key={i} style={{
            background: "rgba(255,255,255,0.04)", border: "0.5px solid rgba(255,255,255,0.08)",
            borderRadius: 10, padding: "8px 12px", display: "flex", alignItems: "center", gap: 8
          }}>
            <div style={{
              width: 26, height: 26, borderRadius: 7, background: c.bg, color: c.color,
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 600
            }}>{c.icon}</div>
            <div>
              <div style={{ fontSize: 12, color: "rgba(240,244,248,0.85)", fontWeight: 500 }}>{c.name}</div>
              <div style={{ fontSize: 11, color: "rgba(240,244,248,0.45)" }}>{c.desc}</div>
            </div>
          </div>
        ))}
      </div>

      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}
