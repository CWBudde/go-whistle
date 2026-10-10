package bench

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
)

const darwin = "darwin"

// Machine is the hardware and software a run measured on.
type Machine struct {
	CPU       string  `json:"cpu"`
	Cores     int     `json:"cores"`
	CoreTypes string  `json:"core_types,omitempty"` // darwin: per performance level, "5 Super + 10 Performance"
	MemoryGiB float64 `json:"memory_gib"`
	OS        string  `json:"os"`
	Arch      string  `json:"arch"`
	GoVersion string  `json:"go_version"`
}

// StartRun describes a run starting now: date, commit, machine and the
// load average. Finish completes it.
func StartRun(ctx context.Context) Run {
	return Run{
		Date: time.Now().Format(time.RFC3339), Commit: Commit(ctx), Machine: Describe(ctx), LoadAvgStart: LoadAverage(ctx),
	}
}

// Finish records the load average at the end and the run's wall time.
func (r *Run) Finish(ctx context.Context) {
	r.LoadAvgEnd = LoadAverage(ctx)

	start, err := time.Parse(time.RFC3339, r.Date)
	if err == nil {
		r.Seconds = time.Since(start).Round(time.Second).Seconds()
	}
}

// Describe is this machine. Fields it cannot read stay empty.
func Describe(ctx context.Context) Machine {
	m := Machine{Cores: runtime.NumCPU(), Arch: runtime.GOARCH, GoVersion: runtime.Version()}

	switch runtime.GOOS {
	case darwin:
		m.CPU = sysctl(ctx, "machdep.cpu.brand_string")
		m.CoreTypes = darwinCoreTypes(ctx)
		m.MemoryGiB = gib(sysctl(ctx, "hw.memsize"), 1)
		m.OS = strings.TrimSpace(fmt.Sprintf("%s %s (%s)", command(ctx, "sw_vers", "-productName"),
			command(ctx, "sw_vers", "-productVersion"), command(ctx, "sw_vers", "-buildVersion")))
	case "linux":
		m.CPU = procField("/proc/cpuinfo", "model name")
		m.MemoryGiB = gib(strings.TrimSuffix(procField("/proc/meminfo", "MemTotal"), " kB"), 1<<10)
		m.OS = strings.Trim(procField("/etc/os-release", "PRETTY_NAME"), `"`) + ", kernel " + command(ctx, "uname", "-r")
	default:
		m.OS = runtime.GOOS
	}

	return m
}

// darwinCoreTypes lists the cores per performance level.
func darwinCoreTypes(ctx context.Context) string {
	n, _ := strconv.Atoi(sysctl(ctx, "hw.nperflevels"))
	parts := make([]string, 0, n)

	for i := range n {
		level := fmt.Sprintf("hw.perflevel%d.", i)
		parts = append(parts, sysctl(ctx, level+"physicalcpu")+" "+sysctl(ctx, level+"name"))
	}

	return strings.Join(parts, " + ")
}

// LoadAverage is the 1, 5 and 15 minute load average, "2.11 2.44 2.23", or
// empty where it cannot be read.
func LoadAverage(ctx context.Context) string {
	switch runtime.GOOS {
	case darwin:
		return strings.Trim(sysctl(ctx, "vm.loadavg"), "{} ")
	case "linux":
		raw, err := os.ReadFile("/proc/loadavg")
		if err != nil {
			return ""
		}

		f := strings.Fields(string(raw))
		if len(f) < 3 {
			return ""
		}

		return strings.Join(f[:3], " ")
	default:
		return ""
	}
}

// Commit is git describe of the working tree, "62a4458a1b" or
// "62a4458a1b-dirty" with uncommitted changes to tracked files, or empty
// outside a checkout.
func Commit(ctx context.Context) string {
	return command(ctx, "git", "describe", "--always", "--dirty", "--abbrev=10", "--exclude=*")
}

func sysctl(ctx context.Context, name string) string { return command(ctx, "sysctl", "-n", name) }

// command runs name with args and returns its trimmed output, empty on
// any failure.
func command(ctx context.Context, name string, args ...string) string {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	out, err := exec.CommandContext(ctx, name, args...).Output() // #nosec G204 -- fixed tools and arguments.
	if err != nil {
		return ""
	}

	return strings.TrimSpace(string(out))
}

// procField is the value of the first "key: value" or "key=value" line of
// a /proc or /etc file.
func procField(path, key string) string {
	raw, err := os.ReadFile(path) // #nosec G304 -- fixed paths.
	if err != nil {
		return ""
	}

	for line := range strings.Lines(string(raw)) {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			k, v, ok = strings.Cut(line, "=")
		}

		if ok && strings.TrimSpace(k) == key {
			return strings.TrimSpace(v)
		}
	}

	return ""
}

// gib is a byte count, given in units of unit bytes, in GiB.
func gib(s string, unit float64) float64 {
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0
	}

	return v * unit / (1 << 30)
}

// MaxRSSBytes is getrusage's ru_maxrss in bytes: darwin reports bytes, the
// other unix systems KiB.
func MaxRSSBytes(maxrss int64) int64 {
	if runtime.GOOS == darwin {
		return maxrss
	}

	return maxrss << 10
}
