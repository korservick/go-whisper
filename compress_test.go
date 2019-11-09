package whisper

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/kr/pretty"
)

func TestBitsReadWrite(t *testing.T) {
	buf := make([]byte, 256)

	var bw bitsWriter
	bw.buf = buf
	bw.bitPos = 7

	var br bitsReader
	br.buf = buf
	br.bitPos = 7

	input := []struct {
		val uint64
		len int
	}{
		{len: 8, val: 5},
		{len: 16, val: 97},
		{len: 32, val: 123},
		{len: 64, val: math.Float64bits(95.1)},
		{len: 64, val: 0xf1f2f3f4f5f6f7f8},

		{len: 1, val: 1},
		{len: 15, val: 0x7f7f},
		{len: 55, val: 0x7f2f3f4f5f6f7f},
	}
	for _, d := range input {
		bw.Write(d.len, d.val)
	}
	for i, d := range input {
		if got, want := br.Read(d.len), d.val; got != want {
			t.Errorf("%d: br.Read(%d) = %x; want %b", i, d.len, got, want)
		}
	}
}

func init() {
	log.SetFlags(log.Lshortfile)
	rand.Seed(time.Now().UnixNano())
}

func TestBlockReadWrite1(t *testing.T) {
	for i := 0; i < 1; i++ {
		var acv archiveInfo
		acv.secondsPerPoint = 1
		acv.numberOfPoints = 64
		acv.cblock.lastByteBitPos = 7
		acv.blockSize = 64 * PointSize
		acv.blockRanges = make([]blockRange, 1)

		ts := 1543689630
		var delta int
		next := func(incs ...int) int {
			for _, i := range incs {
				delta += i
			}
			return ts + delta
		}

		var input []dataPoint
		{
			rand.Seed(time.Now().Unix())
			input = append(input, dataPoint{interval: next(1), value: 1})
			input = append(input, dataPoint{interval: next(1), value: 1})
			input = append(input, dataPoint{interval: next(1), value: 1})
			for i := 0; i < 200; i++ {
				input = append(input, dataPoint{interval: next(rand.Intn(10)), value: rand.NormFloat64()})
			}
		}

		buf := make([]byte, acv.blockSize)
		_, left, _ := acv.AppendPointsToBlock(buf, input)

		points := make([]dataPoint, 0, 200)
		points, _, err := acv.ReadFromBlock(buf, points, ts, ts+60*60*60)
		if err != nil {
			t.Error(err)
		}

		if !reflect.DeepEqual(input, append(points, left...)) {
			if diff := cmp.Diff(input, points, cmp.AllowUnexported(dataPoint{})); diff != "" {
				t.Error(diff)
			}

			t.FailNow()
		}
	}
}

func TestBlockReadWrite2(t *testing.T) {
	for i := 0; i < 1; i++ {
		var acv archiveInfo
		acv.secondsPerPoint = 1
		acv.numberOfPoints = 100
		acv.cblock.lastByteBitPos = 7
		acv.blockSize = int(float32(acv.numberOfPoints) * avgCompressedPointSize)
		acv.blockRanges = make([]blockRange, 1)

		var input []dataPoint = []dataPoint{
			{interval: 1544456874, value: 12},
			{interval: 1544456875, value: 24},
			{interval: 1544456876, value: 15},
			{interval: 1544456877, value: 1},
			{interval: 1544456878, value: 2},
			{interval: 1544456888, value: 3},
			{interval: 1544456889, value: 4},
			{interval: 1544457000, value: 4},
			{interval: 1544458000, value: 4},
			{interval: 1544476889, value: 4},
		}

		buf := make([]byte, acv.blockSize)
		acv.AppendPointsToBlock(buf, input[:1])
		acv.AppendPointsToBlock(buf[acv.cblock.lastByteOffset:], input[1:5])
		acv.AppendPointsToBlock(buf[acv.cblock.lastByteOffset:], input[5:])

		points, _, err := acv.ReadFromBlock(buf, make([]dataPoint, 0, 200), 1544456874, 1544477000)
		if err != nil {
			t.Error(err)
		}

		if !reflect.DeepEqual(input, points) {
			pretty.Printf("%# v\n", input)
			pretty.Printf("%# v\n", points)
			if diff := cmp.Diff(input, points, cmp.AllowUnexported(dataPoint{})); diff != "" {
				t.Error(diff)
			}

			t.FailNow()
		}
	}
}

func TestCompressedWhisperReadWrite1(t *testing.T) {
	fpath := "comp.whisper"
	os.Remove(fpath)
	whisper, err := CreateWithOptions(
		fpath,
		[]*Retention{
			{secondsPerPoint: 1, numberOfPoints: 100},
			{secondsPerPoint: 5, numberOfPoints: 100},
		},
		Sum,
		0.7,
		&Options{Compressed: true, PointsPerBlock: 7200},
	)
	if err != nil {
		panic(err)
	}

	ts := int(Now().Add(time.Second * -60).Unix())
	var delta int
	next := func(incs int) int {
		delta += incs
		return ts + delta
	}
	input := []*TimeSeriesPoint{
		{Time: next(1), Value: 12},
		{Time: next(1), Value: 24},
		{Time: next(1), Value: 15},
		{Time: next(1), Value: 1},
		{Time: next(1), Value: 2},
		{Time: next(10), Value: 3},
		{Time: next(1), Value: 4},
		{Time: next(1), Value: 15.5},
		{Time: next(1), Value: 14.0625},
		{Time: next(1), Value: 3.25},
		{Time: next(1), Value: 8.625},
		{Time: next(1), Value: 13.1},
	}

	if err := whisper.UpdateMany(input); err != nil {
		t.Error(err)
	}
	whisper.Close()

	whisper, err = OpenWithOptions(fpath, &Options{Compressed: true, PointsPerBlock: 7200})
	if err != nil {
		t.Fatal(err)
	}

	expectVals := make([]float64, 60)
	for i := 0; i < 60; i++ {
		expectVals[i] = math.NaN()
	}
	for _, p := range input {
		expectVals[p.Time-ts-1] = p.Value
	}
	expect := &TimeSeries{
		fromTime:  ts + 1,
		untilTime: ts + 61,
		step:      1,
		values:    expectVals,
	}
	if ts, err := whisper.Fetch(ts, ts+300); err != nil {
		t.Error(err)
	} else if diff := cmp.Diff(ts, expect, cmp.AllowUnexported(TimeSeries{}), cmpopts.EquateNaNs()); diff != "" {
		t.Error(diff)
	}
}

func TestCompressedWhisperReadWrite2(t *testing.T) {
	fpath := "comp.whisper"
	os.Remove(fpath)
	whisper, err := CreateWithOptions(
		fpath,
		[]*Retention{
			{secondsPerPoint: 1, numberOfPoints: 100},
			{secondsPerPoint: 5, numberOfPoints: 100},
		},
		Sum,
		0.7,
		&Options{Compressed: true, PointsPerBlock: 7200},
	)
	if err != nil {
		panic(err)
	}

	nowTs := 1544478230
	Now = func() time.Time { return time.Unix(int64(nowTs), 0) }
	defer func() { Now = func() time.Time { return time.Now() } }()

	input := []*TimeSeriesPoint{
		{Time: nowTs - 300, Value: 666},

		{Time: 1544478201, Value: 12},

		{Time: 1544478211, Value: 24},
		{Time: 1544478212, Value: 15},
		{Time: 1544478213, Value: 1},
		{Time: 1544478214, Value: 2},

		{Time: 1544478224, Value: 3},
		{Time: 1544478225, Value: 4},
		{Time: 1544478226, Value: 15.5},
		{Time: 1544478227, Value: 14.0625},
		{Time: 1544478228, Value: 3.25},
		{Time: 1544478229, Value: 8.625},
		{Time: nowTs, Value: 13.1},
	}

	for _, p := range input {
		if err := whisper.UpdateMany([]*TimeSeriesPoint{p}); err != nil {
			t.Error(err)
		}
	}
	whisper.Close()

	whisper, err = OpenWithOptions(fpath, &Options{Compressed: true, PointsPerBlock: 7200})
	if err != nil {
		t.Fatal(err)
	}

	{
		expectVals := make([]float64, 4)
		for i := 0; i < 4; i++ {
			expectVals[i] = math.NaN()
		}
		expectVals[1] = input[0].Value
		expect := &TimeSeries{
			fromTime:  1544477925,
			untilTime: 1544477945,
			step:      5,
			values:    expectVals,
		}
		if ts, err := whisper.Fetch(nowTs-310, nowTs-290); err != nil {
			t.Error(err)
		} else if diff := cmp.Diff(ts, expect, cmp.AllowUnexported(TimeSeries{}), cmpopts.EquateNaNs()); diff != "" {
			t.Error(diff)
		}
	}

	{
		expectVals := make([]float64, 30)
		for i := 0; i < 30; i++ {
			expectVals[i] = math.NaN()
		}
		for _, p := range input[1:] {
			expectVals[29-(nowTs-p.Time)] = p.Value
		}
		expect := &TimeSeries{
			fromTime:  1544478201,
			untilTime: 1544478231,
			step:      1,
			values:    expectVals,
		}
		if ts, err := whisper.Fetch(nowTs-30, nowTs); err != nil {
			t.Error(err)
		} else if diff := cmp.Diff(ts, expect, cmp.AllowUnexported(TimeSeries{}), cmpopts.EquateNaNs()); diff != "" {
			t.Error(diff)
		}
	}
}

var fullTest3 = flag.Bool("full-test3", false, "run a full test of TestCompressedWhisperReadWrite3")

// To run a full test of TestCompressedWhisperReadWrite3, it would take about 10
// minutes, the slowness comes from standard whisper file propagation (around 10
// times slower and comsume much more memory than compressed format).
//
// Parallel is disabled because we need to manipulate Now in order to simulate
// updates.
func TestCompressedWhisperReadWrite3(t *testing.T) {
	// TODO: add a test case of mixing random and sequential values/times
	inputs := []struct {
		name      string
		randLimit func() int
		gen       func(prevTime time.Time, index int) *TimeSeriesPoint
	}{
		{
			name: "random_time",
			gen: func(prevTime time.Time, index int) *TimeSeriesPoint {
				return &TimeSeriesPoint{
					Value: 0,
					Time:  int(prevTime.Add(time.Duration(rand.Intn(4096)+1) * time.Second).Unix()),
				}
			},
		},
		{
			name: "random_time_value",
			gen: func(prevTime time.Time, index int) *TimeSeriesPoint {
				return &TimeSeriesPoint{
					Value: rand.NormFloat64(),
					Time:  int(prevTime.Add(time.Duration(rand.Intn(3600*24)+1) * time.Second).Unix()),
				}
			},
		},
		{
			name: "less_random_time_value",
			gen: func(prevTime time.Time, index int) *TimeSeriesPoint {
				return &TimeSeriesPoint{
					Value: 2000.0 + float64(rand.Intn(1000)),
					Time:  int(prevTime.Add(time.Duration(rand.Intn(60)) * time.Second).Unix()),
				}
			},
		},

		{
			name: "random_value",
			gen: func(prevTime time.Time, index int) *TimeSeriesPoint {
				return &TimeSeriesPoint{
					Value: rand.NormFloat64(),
					Time:  int(prevTime.Add(time.Second).Unix()),
				}
			},
		},
		{
			name:      "random_value2",
			randLimit: func() int { return rand.Intn(300) + (60 * 60 * 24) },
			gen: func(prevTime time.Time, index int) *TimeSeriesPoint {
				return &TimeSeriesPoint{
					Value: 2000.0 + float64(rand.Intn(1000)),
					Time:  int(prevTime.Add(time.Second).Unix()),
				}
			},
		},
		{
			name: "simple",
			gen: func(prevTime time.Time, index int) *TimeSeriesPoint {
				return &TimeSeriesPoint{Value: 0, Time: int(prevTime.Add(time.Second).Unix())}
			},
		},
	}

	os.MkdirAll("tmp", 0755)
	inMemory := true
	for i := range inputs {
		input := inputs[i]
		if input.randLimit == nil {
			input.randLimit = func() int { return rand.Intn(300) }
		}

		t.Run(input.name, func(t *testing.T) {
			// can't run tests parallel here because they modify Now
			// t.Parallel()
			t.Logf("case: %s\n", input.name)

			fpath := fmt.Sprintf("tmp/test3_%s.wsp", input.name)
			os.Remove(fpath)
			os.Remove(fpath + ".cwsp")

			cwhisper, err := CreateWithOptions(
				fpath+".cwsp",
				[]*Retention{
					{secondsPerPoint: 1, numberOfPoints: 172800},   // 1s:2d
					{secondsPerPoint: 60, numberOfPoints: 40320},   // 1m:28d
					{secondsPerPoint: 3600, numberOfPoints: 17520}, // 1h:2y
				},
				Average,
				0,
				&Options{Compressed: true, PointsPerBlock: 7200, InMemory: inMemory},
			)
			if err != nil {
				panic(err)
			}
			ncwhisper, err := CreateWithOptions(
				fpath,
				[]*Retention{
					{secondsPerPoint: 1, numberOfPoints: 172800},   // 1s:2d
					{secondsPerPoint: 60, numberOfPoints: 40320},   // 1m:28d
					{secondsPerPoint: 3600, numberOfPoints: 17520}, // 1h:2y
				},
				Average,
				0,
				&Options{Compressed: false, PointsPerBlock: 7200, InMemory: inMemory},
			)
			if err != nil {
				panic(err)
			}
			cwhisper.Close()
			ncwhisper.Close()

			var now = time.Now()
			var total = 60*60*24*365*2 + 37
			var start = now.Add(time.Second * time.Duration(total) * -1)
			Now = func() time.Time { return start }
			defer func() { Now = func() time.Time { return time.Now() } }()

			// var psArr [][]*TimeSeriesPoint
			var ps []*TimeSeriesPoint
			var limit = input.randLimit()
			var statTotalUpdates, extended, totalPoints int
			firstArchiveBound := cwhisper.Retentions()[0].MaxRetention()
			for i := 0; i < total; i++ {
				p := input.gen(start, i)
				var toAppend = true
				if len(ps) == 0 || p.Time-ps[0].Time < firstArchiveBound {
					ps = append(ps, p)
					start = time.Unix(int64(p.Time), 0)
					toAppend = false
				}

				if toAppend || len(ps) >= limit || start.After(now) {
					// fmt.Printf("%d toAppend = %v\r", start.Unix(), len(ps))
					// fmt.Printf("progress: %.2f%% len(points): %d\r", 100-float64(now.Unix()-start.Unix())*100/float64(total), len(ps))
					limit = input.randLimit()
					statTotalUpdates++
					totalPoints += len(ps)
					// psArr = append(psArr, ps)

					cwhisper, err = OpenWithOptions(fpath+".cwsp", &Options{InMemory: inMemory})
					if err != nil {
						t.Fatal(err)
					}
					if err := cwhisper.UpdateMany(ps); err != nil {
						t.Fatal(err)
					}
					if err := cwhisper.Close(); err != nil {
						t.Fatal(err)
					}

					if cwhisper.Extended {
						// for _, a := range cwhisper.archives {
						// 	t.Logf("extended: %s: %d\n", a.Retention, a.totalPoints())
						// }
						extended++
					}

					if *fullTest3 {
						ncwhisper, err = OpenWithOptions(fpath, &Options{InMemory: inMemory})
						if err != nil {
							t.Fatal(err)
						}
						if err := ncwhisper.UpdateMany(ps); err != nil {
							t.Fatal(err)
						}
						if err := ncwhisper.Close(); err != nil {
							t.Fatal(err)
						}
					}

					ps = ps[:0]
				}
				if start.After(now) {
					break
				}
			}

			t.Logf("statTotalUpdates: %d extended: %d totalPoints: %d\n", statTotalUpdates, extended, totalPoints)
			// for _, a := range cwhisper.archives {
			// 	t.Logf("%s: %d\n", a.Retention, a.totalPoints())
			// }

			if inMemory {
				if err := newMemFile(fpath).dumpOnDisk(fpath); err != nil {
					t.Fatal(err)
				}
				if err := newMemFile(fpath + ".cwsp").dumpOnDisk(fpath + ".cwsp"); err != nil {
					t.Fatal(err)
				}
			}

			// {
			// 	data, err := json.Marshal(psArr)
			// 	if err != nil {
			// 		panic(err)
			// 	}
			// 	if err := ioutil.WriteFile("test3.json", data, 0644); err != nil {
			// 		panic(err)
			// 	}
			// }

			if *fullTest3 {
				t.Log("go", "run", "cmd/verify.go", "-v", "-now", fmt.Sprintf("%d", now.Unix()), fpath, fpath+".cwsp")
				output, err := exec.Command("go", "run", "cmd/verify.go", "-now", fmt.Sprintf("%d", now.Unix()), fpath, fpath+".cwsp").CombinedOutput()
				if err != nil {
					t.Log(string(output))
					t.Error(err)
				}
			}

			std, err := os.Stat(fpath)
			if err != nil {
				t.Error(err)
			}
			cmp, err := os.Stat(fpath + ".cwsp")
			if err != nil {
				t.Error(err)
			}
			t.Logf("compression ratio %s: %.2f%%\n", input.name, float64(cmp.Size()*100)/float64(std.Size()))
		})
	}
}

func TestMixAggregationCompressed(t *testing.T) {
	fpath := "mix.cwsp"
	os.Remove(fpath)

	whisper, err := CreateWithOptions(
		fpath,
		[]*Retention{
			{secondsPerPoint: 1, numberOfPoints: 172800},   // 1s:2d
			{secondsPerPoint: 60, numberOfPoints: 40320},   // 1m:28d
			{secondsPerPoint: 3600, numberOfPoints: 17520}, // 1h:2y
		},
		Mix,
		0,
		&Options{
			Compressed: true, PointsPerBlock: 7200, InMemory: true,
			MixAggregationSpecs: []MixAggregationSpec{
				{Method: Average, Percentile: 0},
				{Method: Sum, Percentile: 0},
				{Method: Last, Percentile: 0},
				{Method: Max, Percentile: 0},
				{Method: Min, Percentile: 0},
				{Method: Percentile, Percentile: 50},
				{Method: Percentile, Percentile: 95},
				{Method: Percentile, Percentile: 99},
			},
		},
	)
	if err != nil {
		panic(err)
	}
	// whisper.Close()

	// var now = time.Now()
	// var total = 60*60*24*365*2 + 37
	var start = time.Now().Add(time.Duration(time.Hour * -24))
	// var start = time.Unix(1491778800, 0)
	Now = func() time.Time { return start }
	defer func() { Now = func() time.Time { return time.Now() } }()

	var ps []*TimeSeriesPoint
	for i := 0; i < 24*60*60; i++ {
		ps = append(ps, &TimeSeriesPoint{
			Time: int(start.Unix()),
			// Value: float64(rand.Intn(100000)),
			Value: float64(i),
		})
		start = start.Add(time.Second)
	}
	if err := whisper.UpdateMany(ps); err != nil {
		t.Fatal(err)
	}
	if err := whisper.Close(); err != nil {
		t.Fatal(err)
	}
	whisper.file.(*memFile).dumpOnDisk(fpath)

	// TODO: make some real data validations
	{
		vals, err := whisper.FetchByAggregation(int(Now().Add(time.Hour*24*-3).Unix()), int(Now().Unix()), &MixAggregationSpec{Method: Last})
		if err != nil {
			t.Error(err)
		}
		log.Printf("vals = %+v\n", vals)
	}
	{
		vals, err := whisper.FetchByAggregation(int(Now().Add(time.Hour*24*-3).Unix()), int(Now().Unix()), &MixAggregationSpec{Method: Percentile, Percentile: 50})
		if err != nil {
			t.Error(err)
		}
		log.Printf("vals = %+v\n", vals)
	}
	{
		vals, err := whisper.FetchByAggregation(int(Now().Add(time.Hour*24*-30).Unix()), int(Now().Unix()), &MixAggregationSpec{Method: Last})
		if err != nil {
			t.Error(err)
		}
		log.Printf("vals = %+v\n", vals)
	}
	{
		vals, err := whisper.FetchByAggregation(int(Now().Add(time.Hour*24*-30).Unix()), int(Now().Unix()), &MixAggregationSpec{Method: Percentile, Percentile: 50})
		if err != nil {
			t.Error(err)
		}
		log.Printf("vals = %+v\n", vals)
	}
}

func TestCompressTo(t *testing.T) {
	fpath := "compress_to.wsp"
	os.Remove(fpath)

	whisper, err := CreateWithOptions(
		fpath,
		[]*Retention{
			{secondsPerPoint: 1, numberOfPoints: 172800},   // 1s:2d
			{secondsPerPoint: 60, numberOfPoints: 40320},   // 1m:28d
			{secondsPerPoint: 3600, numberOfPoints: 17520}, // 1h:2y
		},
		Average,
		0,
		&Options{Compressed: false, PointsPerBlock: 7200, InMemory: true},
	)
	if err != nil {
		panic(err)
	}
	whisper.Close()

	for _, archive := range whisper.archives {
		var ps []*TimeSeriesPoint
		for i := 0; i < archive.numberOfPoints; i++ {
			start := Now().Add(time.Second * time.Duration(archive.secondsPerPoint*i) * -1)
			ps = append(ps, &TimeSeriesPoint{
				// Time: int(start.Add(time.Duration(i) * time.Second).Unix()),
				Time: int(start.Unix()),
				// Value: float64(i),
				// Value: 2000.0 + float64(rand.Intn(100000))/100.0,
				// Value: rand.NormFloat64(),
				Value: float64(rand.Intn(100000)),
			})
		}
		whisper, err = OpenWithOptions(fpath, &Options{InMemory: true})
		if err != nil {
			t.Fatal(err)
		}
		if err := whisper.UpdateMany(ps); err != nil {
			t.Fatal(err)
		}
		if err := whisper.Close(); err != nil {
			t.Fatal(err)
		}
	}
	whisper.file.(*memFile).dumpOnDisk(fpath)

	whisper, err = OpenWithOptions(fpath, &Options{})
	if err != nil {
		t.Fatal(err)
	}
	os.Remove(fpath + ".cwsp")
	if err := whisper.CompressTo(fpath + ".cwsp"); err != nil {
		t.Fatal(err)
	}

	output, err := exec.Command("go", "run", "cmd/verify.go", fpath, fpath+".cwsp").CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %s", err, output)
	}
}

func TestRandomReadWrite(t *testing.T) {
	// os.Remove("test_random_read_write.wsp")
	fileTs := time.Now().Unix()
	cwhisper, err := CreateWithOptions(
		fmt.Sprintf("test_random_read_write.%d.wsp", fileTs),
		[]*Retention{
			{secondsPerPoint: 1, numberOfPoints: 1728000},
			// {secondsPerPoint: 60, numberOfPoints: 40320},
			// {secondsPerPoint: 3600, numberOfPoints: 17520},
		},
		Sum,
		0,
		&Options{Compressed: true, PointsPerBlock: 7200},
	)
	if err != nil {
		panic(err)
	}

	start := Now()
	ptime := start
	var ps []*TimeSeriesPoint
	var vals []float64
	var entropy int
	for i := 0; i < cwhisper.Retentions()[0].numberOfPoints; i++ {
		gap := rand.Intn(10) + 1
		ptime = ptime.Add(time.Second * time.Duration(gap))
		if ptime.After(start.Add(time.Duration(cwhisper.Retentions()[0].numberOfPoints) * time.Second)) {
			break
		}
		for j := gap; j > 1; j-- {
			vals = append(vals, math.NaN())
		}
		ts := &TimeSeriesPoint{
			Time:  int(ptime.Unix()),
			Value: rand.NormFloat64(),
			// Value: 2000.0 + float64(rand.Intn(100000))/100.0,
			// Value: float64(rand.Intn(100000)),
		}
		ps = append(ps, ts)
		vals = append(vals, ts.Value)
		entropy++
	}

	if err := cwhisper.UpdateMany(ps); err != nil {
		t.Fatal(err)
	}

	Now = func() time.Time { return time.Unix(int64(ps[len(ps)-1].Time), 0) }
	defer func() { Now = func() time.Time { return time.Now() } }()

	ts, err := cwhisper.Fetch(int(start.Unix()), int(ptime.Unix()))
	if err != nil {
		t.Fatal(err)
	}

	// log.Printf("entropy = %+v\n", entropy)
	// log.Printf("len(vals) = %+v\n", len(vals))
	// log.Printf("len(ts.Values()) = %+v\n", len(ts.Values()))

	if diff := cmp.Diff(ts.Values(), vals, cmp.AllowUnexported(dataPoint{}), cmpopts.EquateNaNs()); diff != "" {
		// t.Error(diff)
		t.Error("mismatch")
		cache, err := os.Create(fmt.Sprintf("test_random_read_write.%d.json", fileTs))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.NewEncoder(cache).Encode(ps); err != nil {
			t.Fatal(err)
		}
		cache.Close()
	}

	if err := cwhisper.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestFillCompressed(t *testing.T) {
	fpath := "fill.wsp"
	os.Remove(fpath)
	os.Remove(fpath + ".cwsp")

	standard, err := CreateWithOptions(
		fpath,
		[]*Retention{
			{secondsPerPoint: 1, numberOfPoints: 172800},   // 1s:2d
			{secondsPerPoint: 60, numberOfPoints: 40320},   // 1m:28d
			{secondsPerPoint: 3600, numberOfPoints: 17520}, // 1h:2y
		},
		Average,
		0,
		&Options{Compressed: false, PointsPerBlock: 7200, InMemory: true},
	)
	if err != nil {
		panic(err)
	}

	points := []*TimeSeriesPoint{}
	twoYearsAgo := Now().Add(time.Hour * 24 * 365 * -2)
	for i := 0; i < 2*365*24-28*24; i++ {
		points = append(points, &TimeSeriesPoint{
			Time:  int(twoYearsAgo.Add(time.Hour * time.Duration(i)).Unix()),
			Value: rand.NormFloat64(),
		})
	}
	if err := standard.UpdateMany(points); err != nil {
		t.Error(err)
	}

	points = []*TimeSeriesPoint{}
	oneMonthAgo := Now().Add(time.Hour * 24 * -28)
	for i := 0; i < 28*24*60-2*24*60; i++ {
		points = append(points, &TimeSeriesPoint{
			Time:  int(oneMonthAgo.Add(time.Minute * time.Duration(i)).Unix()),
			Value: rand.NormFloat64(),
		})
	}
	if err := standard.UpdateMany(points); err != nil {
		t.Error(err)
	}

	compressed, err := CreateWithOptions(
		fpath+".cwsp",
		[]*Retention{
			{secondsPerPoint: 1, numberOfPoints: 172800, avgCompressedPointSize: 9},   // 1s:2d
			{secondsPerPoint: 60, numberOfPoints: 40320, avgCompressedPointSize: 9},   // 1m:28d
			{secondsPerPoint: 3600, numberOfPoints: 17520, avgCompressedPointSize: 9}, // 1h:2y
		},
		Average,
		0,
		&Options{Compressed: true, PointsPerBlock: 7200, InMemory: true},
	)
	if err != nil {
		panic(err)
	}
	points = []*TimeSeriesPoint{}
	twoDaysAgo := Now().Add(time.Hour * 24 * -2)
	for i := 0; i < 60*60*24*2; i++ {
		points = append(points, &TimeSeriesPoint{
			Time:  int(twoDaysAgo.Add(time.Second * time.Duration(i)).Unix()),
			Value: rand.NormFloat64(),
		})
	}
	if err := compressed.UpdateMany(points); err != nil {
		t.Error(err)
	}

	if err := compressed.file.(*memFile).dumpOnDisk(fpath + ".original.cwsp"); err != nil {
		t.Error(err)
	}

	if err := compressed.FillCompressed(standard); err != nil {
		t.Error(err)
	}

	if err := compressed.file.(*memFile).dumpOnDisk(fpath + ".cwsp"); err != nil {
		t.Error(err)
	}
	if err := standard.file.(*memFile).dumpOnDisk(fpath); err != nil {
		t.Error(err)
	}

	compare := func(w1, w2 *Whisper, from, until int) {
		valsc, err := w1.Fetch(from, until)
		if err != nil {
			t.Error(err)
		}
		valss, err := w2.Fetch(from, until)
		if err != nil {
			t.Error(err)
		}
		var diff, same int
		for i := 0; i < len(valsc.values); i++ {
			vc := valsc.values[i]
			vs := valss.values[i]
			if math.IsNaN(vc) && math.IsNaN(vs) {
				same++
			} else if vc != vs {
				t.Errorf("%d/%d %d: %v != %v\n", i, len(valsc.values), valsc.fromTime+i*valsc.step, vc, vs)
				diff++
			} else {
				same++
			}
		}
		if diff > 0 {
			t.Errorf("diff = %d", diff)
			t.Errorf("same = %d", same)
		}
	}

	t.Log("comparing 2 years archive")
	compare(compressed, standard, int(twoYearsAgo.Unix()), int(Now().Add(time.Hour*24*-28).Unix()))
	t.Log("comparing 1 month archive")
	compare(compressed, standard, int(oneMonthAgo.Add(time.Hour).Unix()), int(Now().Add(time.Hour*24*-2-time.Hour).Unix()))

	oldCompressed, err := OpenWithOptions(fpath+".original.cwsp", &Options{})
	if err != nil {
		t.Error(err)
	}
	t.Log("comparing 2 days archive")
	compare(compressed, oldCompressed, int(Now().Add(time.Hour*24*-2+time.Hour).Unix()), int(Now().Unix()))
}

func TestSanitizeAvgCompressedPointSizeOnCreate(t *testing.T) {
	var cases = []struct {
		rets   []*Retention
		expect float32
	}{
		{
			rets: []*Retention{
				{secondsPerPoint: 1, numberOfPoints: 100, avgCompressedPointSize: math.Float32frombits(0xffc00000)}, // 32bits nan
				{secondsPerPoint: 5, numberOfPoints: 100},
			},
			expect: avgCompressedPointSize,
		},
		{
			rets: []*Retention{
				{secondsPerPoint: 1, numberOfPoints: 100, avgCompressedPointSize: float32(math.NaN())}, // 62 bits nan
				{secondsPerPoint: 5, numberOfPoints: 100},
			},
			expect: avgCompressedPointSize,
		},
		{
			rets: []*Retention{
				{secondsPerPoint: 1, numberOfPoints: 100, avgCompressedPointSize: 0},
				{secondsPerPoint: 5, numberOfPoints: 100},
			},
			expect: avgCompressedPointSize,
		},
		{
			rets: []*Retention{
				{secondsPerPoint: 1, numberOfPoints: 100, avgCompressedPointSize: -10},
				{secondsPerPoint: 5, numberOfPoints: 100},
			},
			expect: avgCompressedPointSize,
		},
		{
			rets: []*Retention{
				{secondsPerPoint: 1, numberOfPoints: 100, avgCompressedPointSize: 65536},
				{secondsPerPoint: 5, numberOfPoints: 100},
			},
			expect: MaxCompressedPointSize,
		},
	}
	for _, c := range cases {
		fpath := "extend.whisper"
		os.Remove(fpath)
		whisper, err := CreateWithOptions(
			fpath,
			c.rets,
			Sum,
			0.7,
			&Options{Compressed: true, PointsPerBlock: 7200},
		)
		if err != nil {
			panic(err)
		}
		whisper.Close()

		whisper, err = OpenWithOptions(fpath, &Options{Compressed: true, PointsPerBlock: 7200})
		if err != nil {
			t.Fatal(err)
		}
		if got, want := whisper.archives[0].avgCompressedPointSize, c.expect; got != want {
			t.Errorf("whisper.archives[0].avgCompressedPointSize = %f; want %f", got, want)
		}
	}
}

func TestEstimatePointSize(t *testing.T) {
	cases := []struct {
		input  []dataPoint
		expect float32
	}{
		// 0 datapoints
		{input: []dataPoint{}, expect: avgCompressedPointSize},
		// not enough datapoints
		{
			input: []dataPoint{
				{interval: 1543449600, value: 5},
				{interval: 1543478400, value: 5},
			},
			expect: avgCompressedPointSize,
		},
	}
	for _, c := range cases {
		// not enough datapoints
		size := estimatePointSize(c.input, &Retention{secondsPerPoint: 10, numberOfPoints: 17280}, DefaultPointsPerBlock)
		if got, want := size, c.expect; got != want {
			t.Errorf("size = %f; want %f", got, want)
		}
	}

	// for i := 0; i < 500; i += 10 {
	// 	var ds []dataPoint
	// 	var start = 1543449600
	// 	for j := 0; j < i; j++ {
	// 		ds = append(ds, dataPoint{interval: start, value: rand.NormFloat64()})
	// 		// ds = append(ds, dataPoint{interval: start, value: 10})
	//
	// 		start += 1
	// 		// start += rand.Int()
	// 	}
	// 	size := estimatePointSize(ds, &Retention{secondsPerPoint: 10, numberOfPoints: 17280}, DefaultPointsPerBlock)
	// 	fmt.Printf("%d: %f\n", i, size)
	// }

	return
}

func TestFillCompressedByMix(t *testing.T) {
	srcPath := "fill-mix.src.cwsp"
	dstPath := "fill-mix.dst.cwsp"
	os.Remove(srcPath)
	os.Remove(dstPath)

	srcMix, err := CreateWithOptions(
		srcPath,
		[]*Retention{
			{secondsPerPoint: 1, numberOfPoints: 172800},   // 1s:2d
			{secondsPerPoint: 60, numberOfPoints: 40320},   // 1m:28d
			{secondsPerPoint: 3600, numberOfPoints: 17520}, // 1h:2y
		},
		Mix,
		0,
		&Options{
			Compressed: true, PointsPerBlock: 7200, InMemory: true,
			MixAggregationSpecs: []MixAggregationSpec{
				{Method: Average, Percentile: 0},
				{Method: Sum, Percentile: 0},
				{Method: Last, Percentile: 0},
				{Method: Max, Percentile: 0},
				{Method: Min, Percentile: 0},
				{Method: Percentile, Percentile: 50},
				{Method: Percentile, Percentile: 95},
				{Method: Percentile, Percentile: 99},
			},
		},
	)
	if err != nil {
		panic(err)
	}

	points := []*TimeSeriesPoint{}
	twoYearsAgo := Now().Add(time.Hour * 24 * 365 * -2)
	for i := 0; i < 2*365*24-28*24; i++ {
		points = append(points, &TimeSeriesPoint{
			Time:  int(twoYearsAgo.Add(time.Hour * time.Duration(i)).Unix()),
			Value: rand.NormFloat64(),
		})
	}
	if err := srcMix.UpdateMany(points); err != nil {
		t.Error(err)
	}

	points = []*TimeSeriesPoint{}
	oneMonthAgo := Now().Add(time.Hour * 24 * -28)
	for i := 0; i < 28*24*60-2*24*60; i++ {
		points = append(points, &TimeSeriesPoint{
			Time:  int(oneMonthAgo.Add(time.Minute * time.Duration(i)).Unix()),
			Value: rand.NormFloat64(),
		})
	}
	if err := srcMix.UpdateMany(points); err != nil {
		t.Error(err)
	}

	dstMix, err := CreateWithOptions(
		dstPath,
		[]*Retention{
			{secondsPerPoint: 1, numberOfPoints: 172800},   // 1s:2d
			{secondsPerPoint: 60, numberOfPoints: 40320},   // 1m:28d
			{secondsPerPoint: 3600, numberOfPoints: 17520}, // 1h:2y
		},
		Mix,
		0,
		&Options{
			Compressed: true, PointsPerBlock: 7200, InMemory: true,
			MixAggregationSpecs: []MixAggregationSpec{
				{Method: Average, Percentile: 0},
				{Method: Sum, Percentile: 0},
				{Method: Last, Percentile: 0},
				{Method: Max, Percentile: 0},
				{Method: Min, Percentile: 0},
				{Method: Percentile, Percentile: 50},
				{Method: Percentile, Percentile: 95},
				{Method: Percentile, Percentile: 99},
			},
		},
	)
	if err != nil {
		panic(err)
	}
	points = []*TimeSeriesPoint{}
	twoDaysAgo := Now().Add(time.Hour * 24 * -2)
	for i := 0; i < 60*60*24*2; i++ {
		points = append(points, &TimeSeriesPoint{
			Time:  int(twoDaysAgo.Add(time.Second * time.Duration(i)).Unix()),
			Value: rand.NormFloat64(),
		})
	}
	if err := dstMix.UpdateMany(points); err != nil {
		t.Error(err)
	}

	if err := dstMix.file.(*memFile).dumpOnDisk(dstPath + ".bak"); err != nil {
		t.Error(err)
	}

	if err := dstMix.FillCompressed(srcMix); err != nil {
		t.Error(err)
	}

	if err := dstMix.file.(*memFile).dumpOnDisk(dstPath); err != nil {
		t.Error(err)
	}
	if err := srcMix.file.(*memFile).dumpOnDisk(srcPath); err != nil {
		t.Error(err)
	}

	compare := func(w1, w2 *Whisper, from, until int) {
		valsc, err := w1.Fetch(from, until)
		if err != nil {
			t.Error(err)
		}
		valss, err := w2.Fetch(from, until)
		if err != nil {
			t.Error(err)
		}
		var diff, same int
		for i := 0; i < len(valsc.values); i++ {
			vc := valsc.values[i]
			vs := valss.values[i]
			if math.IsNaN(vc) && math.IsNaN(vs) {
				same++
			} else if vc != vs {
				t.Errorf("%d/%d %d: %v != %v\n", i, len(valsc.values), valsc.fromTime+i*valsc.step, vc, vs)
				diff++
			} else {
				same++
			}
		}
		if diff > 0 {
			t.Errorf("diff = %d", diff)
			t.Errorf("same = %d", same)
		}
	}

	t.Log("comparing 2 years archive")
	compare(dstMix, srcMix, int(twoYearsAgo.Unix()), int(Now().Add(time.Hour*24*-28).Unix()))
	t.Log("comparing 1 month archive")
	compare(dstMix, srcMix, int(oneMonthAgo.Add(time.Hour).Unix()), int(Now().Add(time.Hour*24*-2-time.Hour).Unix()))

	oldDstMix, err := OpenWithOptions(dstPath+".bak", &Options{})
	if err != nil {
		t.Error(err)
	}
	t.Log("comparing 2 days archive")
	compare(dstMix, oldDstMix, int(Now().Add(time.Hour*24*-2+time.Hour).Unix()), int(Now().Unix()))
}

func BenchmarkWriteCompressed(b *testing.B) {
	fpath := "benchmark_write.cwsp"
	os.Remove(fpath)
	cwhisper, err := CreateWithOptions(
		fpath,
		[]*Retention{
			{secondsPerPoint: 1, numberOfPoints: 172800},   // 1s:2d
			{secondsPerPoint: 60, numberOfPoints: 40320},   // 1m:28d
			{secondsPerPoint: 3600, numberOfPoints: 17520}, // 1h:2y
		},
		Sum,
		0,
		&Options{Compressed: true, PointsPerBlock: 7200},
	)
	if err != nil {
		b.Fatal(err)
	}

	start := Now()
	var ps, history []*TimeSeriesPoint
	for i := 0; i < b.N; i++ {
		p := &TimeSeriesPoint{
			Time:  int(start.Add(time.Duration(i) * time.Second).Unix()),
			Value: rand.NormFloat64(),
		}
		history = append(history, p)
		ps = append(ps, p)

		if len(ps) >= 300 {
			if err := cwhisper.UpdateMany(ps); err != nil {
				b.Fatal(err)
			}
			ps = ps[:0]
		}
	}
	if err := cwhisper.Close(); err != nil {
		b.Fatal(err)
	}
	td, err := os.Create("test_data")
	if err != nil {
		b.Fatal(err)
	}
	if err := json.NewEncoder(td).Encode(history); err != nil {
		b.Fatal(err)
	}
	if err := td.Close(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkReadCompressed(b *testing.B) {
	fpath := "benchmark_write.cwsp"
	cwhisper, err := OpenWithOptions(fpath, &Options{})
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
		_, err := cwhisper.Fetch(int(time.Now().Add(-48*time.Hour).Unix()), int(time.Now().Unix()))
		if err != nil {
			b.Fatal(err)
		}
	}
	if err := cwhisper.Close(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkReadStandard(b *testing.B) {
	fpath := "benchmark_write.wsp"
	cwhisper, err := OpenWithOptions(fpath, &Options{})
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
		_, err := cwhisper.Fetch(int(time.Now().Add(-48*time.Hour).Unix()), int(time.Now().Unix()))
		if err != nil {
			b.Fatal(err)
		}
	}
	if err := cwhisper.Close(); err != nil {
		b.Fatal(err)
	}
}

func BenchmarkWriteStandard(b *testing.B) {
	fpath := "benchmark_write.wsp"
	os.Remove(fpath)
	cwhisper, err := CreateWithOptions(
		fpath,
		[]*Retention{
			{secondsPerPoint: 1, numberOfPoints: 172800},   // 1s:2d
			{secondsPerPoint: 60, numberOfPoints: 40320},   // 1m:28d
			{secondsPerPoint: 3600, numberOfPoints: 17520}, // 1h:2y
		},
		Sum,
		0,
		&Options{Compressed: false, PointsPerBlock: 7200},
	)
	if err != nil {
		b.Fatal(err)
	}

	start := Now()
	var ps []*TimeSeriesPoint
	for i := 0; i < b.N; i++ {
		ps = append(ps, &TimeSeriesPoint{
			Time:  int(start.Add(time.Duration(i) * time.Second).Unix()),
			Value: rand.NormFloat64(),
		})

		if len(ps) >= 300 {
			if err := cwhisper.UpdateMany(ps); err != nil {
				b.Fatal(err)
			}
			ps = ps[:0]

		}
	}
	if err := cwhisper.Close(); err != nil {
		b.Fatal(err)
	}
}
