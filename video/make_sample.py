"""Builds a sample MacGym demo video (scratch VO) from real recordings + the deck render."""
import asyncio, json, os, subprocess, edge_tts
from PIL import Image, ImageDraw, ImageFont
FF = "/opt/homebrew/bin/ffmpeg"; FP = "/opt/homebrew/bin/ffprobe"
R = "/Users/shlok/macgym/web"; B = "build"
FONT = os.path.expanduser("~/Library/Fonts/Geist-SemiBold.otf")
MONO = os.path.expanduser("~/Library/Fonts/GeistMono-Medium.otf")
TRIPOD4 = f"{R}/deck/data/playground_tripod_4x.mp4"; EDIT = f"{R}/deck/data/playground_edit.mp4"
DECK = f"{R}/deck2/export/macgym-deck.mp4"
VOICE = "en-US-AndrewNeural"
# (source, start, dur, big caption, small tag, voice line, is_placeholder)
SEGS = [
 # ---- PART 1: SLIDES ----
 (DECK, 0.0, 5.5, "", "OPTIONAL: YOUR ON-CAMERA INTRO", "I'm Shlok, and this is MacGym.", True),
 (DECK, 5.5, 8.0, "", "", "Every laptop here has a GPU. For robots, it sits idle, because the fast robot simulators need NVIDIA.", False),
 (DECK, 13.5, 9.0, "", "", "So I built MacGym. You write a robot in Python, and your Mac trains four thousand copies of it at once.", False),
 (DECK, 22.5, 10.0, "", "", "And that's the theme. Elevate the ordinary. The laptop already on your desk becomes a robot learning lab. A GPU that sits idle trains four thousand robots at once.", False),
 (DECK, 32.5, 9.5, "", "", "You describe it. MacGym compiles it into one GPU program, one thread per robot, and it learns to walk.", False),
 (DECK, 42.0, 10.5, "", "", "From falling to running in one minute. Five out of five training runs learned.", False),
 (DECK, 52.5, 8.5, "", "", "Three days of practice, squeezed into twenty eight seconds.", False),
 (DECK, 61.0, 9.5, "", "", "Checked against the pro simulator. Twenty two times faster than MuJoCo on the same robot, it still walks inside MuJoCo, and when my first robots cheated, we caught it and fixed the physics.", False),
 (DECK, 70.5, 9.5, "", "", "And it's a real library. Clone to a trained robot in eighty seven seconds.", False),
 # ---- PART 2: THE SOFTWARE RUNNING ----
 (EDIT, 0.6, 14.5, "Live: add a 4th leg, Cmd+Enter", "real recording · Apple M3 Pro", "Here it is running. This is the playground. I give the robot a fourth leg and press command enter. It recompiles into one Metal kernel.", False),
 (TRIPOD4, 0.0, 15.5, "4,096 copies learning at once", "4x speed · real run", "And four thousand copies start learning at once. Watch them go from flopping around to walking.", False),
 (DECK, 80.0, 11.0, "", "OPTIONAL: YOUR ON-CAMERA CLOSER", "MacGym. Robot learning on the laptop you already own. Elevate the ordinary.", True),
]
def run(c): subprocess.run(c, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
def dur(f): return float(subprocess.check_output([FP,"-v","error","-show_entries","format=duration","-of","csv=p=0",f]).strip())
def overlay_png(i, cap, tag, ph):
    im = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    if cap:
        f = ImageFont.truetype(FONT, 64); d.rectangle([0, 890, 1920, 1080], fill=(17, 17, 17, 210))
        w = d.textlength(cap, font=f); d.text(((1920 - w) / 2, 948), cap, font=f, fill=(255, 255, 255, 255))
    if tag:
        f = ImageFont.truetype(MONO, 30); w = d.textlength(tag, font=f)
        col = (255, 77, 0, 235) if ph else (17, 17, 17, 235)
        d.rectangle([60, 60, 60 + w + 36, 60 + 66], fill=col); d.text((78, 74), tag, font=f, fill=(255, 255, 255, 255))
    p = f"{B}/ov{i}.png"; im.save(p); return p
def esc(t): return t.replace("\\","\\\\").replace(":", "\\:").replace("'", "\u2019").replace(",", "\\,")
async def tts(i, line):
    p = f"{B}/vo{i}.mp3"; await edge_tts.Communicate(line, VOICE, rate="+6%").save(p); return p
async def main():
    os.makedirs(B, exist_ok=True); parts = []
    vos = await asyncio.gather(*[tts(i, s[5]) for i, s in enumerate(SEGS)])
    for i, (src, ss, d, cap, tag, line, ph) in enumerate(SEGS):
        d = max(d, dur(vos[i]) + 0.6)
        vf = ["scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0xFAFAF7", "fps=30"]
        ov = overlay_png(i, cap, tag, ph)
        out = f"{B}/seg{i}.mp4"
        run([FF,"-y","-ss",str(ss),"-t",f"{d:.2f}","-i",src,"-i",vos[i],"-loop","1","-i",ov,"-filter_complex",
             f"[0:v]{','.join(vf)}[b];[b][2:v]overlay=0:0:shortest=1[v];[1:a]adelay=300|300,apad,atrim=0:{d:.2f},asetpts=N/SR/TB[a]",
             "-map","[v]","-map","[a]","-c:v","libx264","-preset","medium","-crf","20","-pix_fmt","yuv420p",
             "-c:a","aac","-ar","48000","-ac","2","-b:a","160k","-shortest",out])
        parts.append(out)
    with open(f"{B}/list.txt","w") as f: f.writelines(f"file '{os.path.basename(p)}'\n" for p in parts)
    run([FF,"-y","-f","concat","-safe","0","-i",f"{B}/list.txt","-c","copy","-movflags","+faststart","macgym-sample.mp4"])
    print("done", dur("macgym-sample.mp4"))
asyncio.run(main())
