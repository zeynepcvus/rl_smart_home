import { useEffect, useState } from "react";

export default function Welcome({ goTo }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => setVisible(true), []);

  return (
    <div style={{
      minHeight: "100vh", background: "#0a1628", display: "flex",
      flexDirection: "column", alignItems: "center", justifyContent: "center",
      padding: "3rem 2rem", position: "relative", overflow: "hidden",
      fontFamily: "'DM Sans', sans-serif"
    }}>
      {/* Grid arka plan */}
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage: "linear-gradient(rgba(29,158,117,0.07) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.07) 1px, transparent 1px)",
        backgroundSize: "40px 40px", pointerEvents: "none"
      }} />

      {/* Glow */}
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
        <div style={{
          width: 6, height: 6, borderRadius: "50%", background: "#1D9E75",
          animation: "blink 2s ease infinite"
        }} />
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
      <button
        onClick={() => goTo("schedule")}
        style={{
          background: "#1D9E75", color: "#fff", border: "none", borderRadius: 10,
          padding: "13px 36px", fontSize: 14, fontWeight: 500, cursor: "pointer",
          display: "flex", alignItems: "center", gap: 10, position: "relative",
          opacity: visible ? 1 : 0, transition: "opacity 0.7s ease 0.3s, background 0.2s"
        }}
        onMouseEnter={e => e.currentTarget.style.background = "#0F6E56"}
        onMouseLeave={e => e.currentTarget.style.background = "#1D9E75"}
      >
        Kuruluma Başla
        <span style={{
          width: 18, height: 18, borderRadius: "50%",
          background: "rgba(255,255,255,0.2)", display: "flex",
          alignItems: "center", justifyContent: "center", fontSize: 11
        }}>→</span>
      </button>

      {/* İstatistikler */}
      <div style={{
        display: "flex", gap: "2rem", marginTop: "3rem", position: "relative",
        opacity: visible ? 1 : 0, transition: "opacity 0.7s ease 0.4s"
      }}>
        {[
          { val: "%4.06", label: "maliyet tasarrufu" },
          { val: "%38", label: "daha az konfor ihlali" },
          { val: "100", label: "senaryo üzerinde test" },
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

      {/* Cihaz kartları */}
      <div style={{
        display: "flex", gap: 10, marginTop: "2.5rem", position: "relative",
        opacity: visible ? 1 : 0, transition: "opacity 0.7s ease 0.5s"
      }}>
        {[
          { icon: "AC", bg: "rgba(29,158,117,0.15)", color: "#5DCAA5", name: "HVAC", desc: "akıllı ısı kontrolü" },
          { icon: "WM", bg: "rgba(55,138,221,0.15)", color: "#85B7EB", name: "Çamaşır", desc: "ucuz saate kaydır" },
          { icon: "LT", bg: "rgba(250,199,117,0.15)", color: "#FAC775", name: "Aydınlatma", desc: "ihtiyaca göre aç/kapat" },
        ].map((c, i) => (
          <div key={i} style={{
            background: "rgba(255,255,255,0.04)", border: "0.5px solid rgba(255,255,255,0.08)",
            borderRadius: 10, padding: "10px 14px", display: "flex", alignItems: "center", gap: 8
          }}>
            <div style={{
              width: 28, height: 28, borderRadius: 7, background: c.bg, color: c.color,
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 500
            }}>{c.icon}</div>
            <div>
              <div style={{ fontSize: 12, color: "rgba(240,244,248,0.85)", fontWeight: 500 }}>{c.name}</div>
              <div style={{ fontSize: 12, color: "rgba(240,244,248,0.55)" }}>{c.desc}</div>
            </div>
          </div>
        ))}
      </div>

      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}