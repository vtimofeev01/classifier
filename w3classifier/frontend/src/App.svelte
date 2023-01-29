<script>
    import {onMount} from "svelte";
    import Box from "./Box.svelte"
    import Image from './Image.svelte'
    // import CBtn from "./CBtn.svelte"

    let source_server = 'http://0.0.0.0:8081'
    let ims = {}            // ret dict
    let imslen = 0

    let filter_label = 'none'        //
    let filter_value = 'all'
    let filter_values = []
    let filter_size = 'none'  // up low
    let filter_textes = []
    let filter_text = 'none'

    let counts_table = ''

    let seek_label = 'none'
    let seek_only_clear = 'no'
    let seekvalues = []
    let seekvalue = ''
    let seekrec = {}
    let seek_icons = {}
    let seek_resp = {}

    let c_folder = 'all'
    let c_folders_show = true
    let c_folders = []


    let index = 0
    let images = []
    let all_labels = []
    let label_data = {}

    let key;
    let keyCode;
    let nums = [96, 97, 98, 99, 100, 101, 102, 103, 104, 105]
    let nums2 = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]

    let storetext = 'store:'
    let store_rec = {}

    async function read_label() {
        const res = await fetch(`${source_server}/get_label_value_on_image/${seek_label}/${images[index]}`)
        seekrec = await res.json()
        seekvalue = seekrec.imlabel
        seek_icons = seekrec.icons
        console.log('seekrec=', seekrec)
        console.log('seekrec icons=', seekrec.icons)
        console.log('seekrec icons2=', seek_icons)
    }

    function on_seek() {
        console.log('seek')
        read_label()
    }


    function handleKeydown(event) {
        key = event.key;
        keyCode = event.keyCode;
        console.log('key=', key, ' keycode=', keyCode)
        if ((keyCode === 37) && (index > 0)) {
            index--;
            on_seek()
        }
        if ((keyCode === 39) && (index < imslen - 1)) {
            index++;
            on_seek()
        }
        if ((keyCode === 32) && (index < imslen)) {
            index++;
            on_seek()
        }
        if (nums.includes(keyCode) && ((keyCode - 96) < seekvalues.length)) {
            seekvalue = seekvalues[keyCode - 96];
            set_new_label()
        }
        if (nums2.includes(keyCode) && ((keyCode - 48) < seekvalues.length)) {
            seekvalue = seekvalues[keyCode - 48];
            set_new_label()
        }
        if ((keyCode === 83) && (seek_label !== 'none')) {
            seek_store()
        }
    }


    async function loadData() {
        console.log('in', filter_label, filter_value, seek_label)
        const res = await fetch(`${source_server}/set_filter/${filter_label}/${filter_value}/${seek_label}/${seek_only_clear}/${filter_size}/${filter_text}/${c_folder}`);
        ims = await res.json()
        console.log('ims', ims)
        images = ims.images
        imslen = images.length
        index = 0
        all_labels = ims.labels
        label_data = ims.label
        filter_values = ims.values
        filter_values.push('to_check')
        // filter_values.push('all')
        filter_textes = ims.text
        c_folders = ['all']
        c_folders = c_folders.concat(ims.folders)
        // c_folders = [..., ...[ims.folders]]
        console.log('filtertextes =', filter_textes)
        seekvalues = ims.seekvalues
        counts_table = ims.counts
        console.log("labelvalues =", filter_values)
        console.log('set filter')
        read_label()
    }

    onMount(async () => {
        await loadData()
    });

    async function set_new_label() {
        const res = await fetch(`${source_server}/set_value/${images[index]}/${seek_label}/${seekvalue}`)
        seek_resp = await res.json()
        if (seek_resp.res !== 'ok') {
            alert(seek_resp.res)
        }
        console.log('set_new_label=', seek_resp)
        console.log(seek_resp)
    }

    async function seek_store() {
        console.log(`storetext ${source_server}/store_label/${seek_label}`)
        const res = await fetch(`${source_server}/store_label/${seek_label}`)
        store_rec = await res.json()
        storetext = store_rec.res
        console.log('storetext:', storetext, 'label:', seek_label)
        setTimeout(function () {
            storetext = 'store';
            console.log('storetext finish')
        }, 3000)
    }

    console.log('started')


</script>
<style>

    .label {
        background-color: #bbbcc2;
    }

    .active {
        background-color: #ff3e00;
        color: white;
        font-weight: bold;
        font-size: x-large;
    }

    .selected {
        background-color: green;
        color: white;
    }

    img {
        height: 50vh;
    }

    div {
        display: flex;
        height: 100%;
        align-items: center;
        justify-content: center;
        flex-direction: column;
    }

    kbd {
        background-color: #eee;
        border-radius: 4px;
        font-size: 6em;
        padding: 0.2em 0.5em;
        border-top: 5px solid rgba(255, 255, 255, 0.5);
        border-left: 5px solid rgba(255, 255, 255, 0.5);
        border-right: 5px solid rgba(0, 0, 0, 0.2);
        border-bottom: 5px solid rgba(0, 0, 0, 0.2);
        color: #555;
    }

    button {
        margin: 0 0 0 0
    }

    table, td, tr {
        border-width: 0;
    }

    .but_sel {
        position: relative;
        display: inline-block;
    }

    .upleftgtay {
        background: rgba(0, 0, 0, 0.7);
        /*background: #222222;*/
        color: #ffffff;
        border-radius: 2px;
        position: absolute;
        left: 0;
        top: 10px;
        font-size: 18px;
        padding: 3px 5px;
    }

        .downleftgtay {
        background: rgba(0, 0, 0, 0.7);
        /*background: #222222;*/
        color: #ffffff;
        border-radius: 2px;
        position: absolute;
        right: 0;
        bottom: 0;
        /*top: 10px;*/
        font-size: 18px;
        padding: 3px 5px;
    }


    .button_im {
        max-width: 90%;
        height: 90%;
        margin: 5px;
    }

    .button_icon {
        max-width: 250px;
        height: 250px;
    }

</style>

<svelte:window on:keydown={handleKeydown}/>

<!--<img src="http://0.0.0.0:8081/marked_image/0125__11_14_00__i0643800__gcn0028419__0028419__person_32206048888487.png" alt="0">-->
<!--<Image src="http://0.0.0.0:8081/marked_image/2803313__0720__01_16_14__________.png" alt="111" />-->

<Box>
    {#if seek_label === 'none'}<h3>select label to edit</h3>{/if}
    <button disabled='{seek_label === "none"}' on:click="{() => {seek_store()}}">{storetext}:{seek_label}</button>
    {#each all_labels as ilabel}
        {#if ilabel === seek_label || seek_label === 'none'}
            <button class="label" class:selected="{seek_label === ilabel}"
                    on:click="{() => {seek_label = (seek_label === 'none')? ilabel:'none';loadData()}}">{ilabel}</button>
        {/if}
    {/each}
    <button class:selected="{seek_only_clear === 'yes'}" on:click="{() => {seek_only_clear = 'yes';loadData()}}">new
    </button>
    <button class:selected="{seek_only_clear === 'no'}" on:click="{() => {seek_only_clear = 'no';loadData()}}">all
    </button>
    <br>
</Box>
<Box>
    <table>
        <tr>
            <td><img src="{source_server}/marked_image/{images[index]}" alt="{index}"/><br></td>
            <td>

                {#each seekvalues as sv, nn}
                    <div class="but_sel">
                        <button class:active="{seekvalue === sv}" on:click="{() => {seekvalue = sv; set_new_label()}}"
                                class="button_icon">
                            <!--                        <div class="but_sel">-->
                            {#if typeof seek_icons[sv] !== 'undefined'}
                                <span class="upleftgtay">{(100 * seekrec['icons'][sv]['thr']).toFixed(1)}%</span>
                                <img class='button_im' src="{source_server}/marked_image/{seek_icons[sv]['image']}"
                                     alt="{seekrec['icons'][sv]['image']}"><br>
                            {/if}
                            <span class="downleftgtay"> ({nn}) {sv}</span>
                            <!--                        </div>-->
                        </button>
                    </div>
                {/each}<br>
                <button class:active="{seekvalue === 'DELETE'}"
                        on:click="{() => {seekvalue = 'DELETE'; set_new_label()}}">Del
                </button>
                <br>
                {#if filter_value === 'to_check'}
                    <br>{filter_textes[index]}
                {/if}
            </td>
        </tr>
    </table>


    <button disabled='{index === 0}' on:click="{() => {index--; on_seek()}}">prev</button>
    {#if imslen > 100}
        <button disabled='{index < 100}' on:click="{() => {index -= 100; on_seek()}}">-100</button>
    {/if}
    {#if imslen > 1000}
        <button disabled='{index < 1000}' on:click="{() => {index -= 1000; on_seek()}}">-1000</button>
    {/if}
    {index + 1} / {imslen}
    {#if imslen > 1000}
        <button disabled='{index > imslen - 1000}' on:click="{() => {index += 1000; on_seek()}}">+1000</button>
    {/if}
    {#if imslen > 100}
        <button disabled='{index > imslen - 100}' on:click="{() => {index += 100; on_seek()}}">+100</button>
    {/if}
    <button disabled='{index === imslen - 1}' on:click="{() => {index++; on_seek()}}">next</button>
</Box>

<!--Filter area -->
<!--Filter area -->
<!--Filter area -->

<Box>
    <h3>select labels filter: total {images.length} imgs</h3>
    <Box cls="thin_box">

        {#each all_labels as ilabel}
            {#if ilabel === filter_label || filter_label === 'none'}
                <button class="label" class:selected="{filter_label === ilabel}"
                        on:click="{() => {filter_label = (filter_label === 'none')?ilabel:'none'; filter_value = 'all'; loadData()}}">{ilabel}</button>
            {/if}
        {/each}


    </Box>
    <!--    <br>-->
    {#if filter_label !== 'none'}

        <Box cls="thin_box">

            {#each filter_values as lv}
                {#if filter_value === lv || filter_value === 'all' || lv === 'all'}
                    <button class:selected="{filter_value === lv}"
                            on:click="{() => {filter_value = (filter_value === 'all')?lv:'all'; loadData()}}">{lv}</button>
                {/if}
            {/each}


        </Box>
    {/if}
    {#if filter_value === 'to_check'}
        <Box cls="thin_box">
            {#each Array.from(new Set(filter_textes)).sort() as ft}
                <button class:selected="{filter_text === ft}"
                        on:click="{() => {filter_text = (filter_text === 'none')?ft:'none'; loadData()}}">{ft}</button>
            {/each}

        </Box>
    {/if}

    <Box cls="thin_box">
        {#each ['height', 'up', 'low', 'small'] as lv}
            {#if filter_size === 'none' || filter_size === lv || lv === 'none'}
                <button class:selected="{filter_size === lv}"
                        on:click="{() => {filter_size = (filter_size==='none')?lv:'none'; loadData()}}">{lv}</button>
            {/if}
        {/each}
    </Box>
    <Box cls="thin_box">
        {#each c_folders as folder}
            {#if c_folders_show || c_folder === folder}
                <button class:selected="{c_folder === folder}"
                        on:click="{() => {c_folder = folder; c_folders_show = !c_folders_show ;  loadData()}}">{folder}</button>
            {/if}
        {/each}
    </Box>

</Box>
<Box>
    {@html counts_table}
</Box>
